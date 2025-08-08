import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import settings
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the FastAPI application
app = FastAPI(
    title="Policy Intent Scoring API",
    description="API for scoring policy intent in press releases",
    version="1.0.0",
)

# Templates
templates = Jinja2Templates(directory="api/templates")

# Global model variable
model: Optional[object] = None


class PolicyIntentRequest(BaseModel):
    """Pydantic model for the request body."""

    headline: str = Field(
        ..., min_length=1, max_length=1000, description="Press release headline"
    )
    body: str = Field(
        ..., min_length=1, max_length=10000, description="Press release body text"
    )


class PolicyIntentResponse(BaseModel):
    """Pydantic model for the response."""

    intent_score: float = Field(..., description="Intent score from 0-100")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    model_loaded: bool
    redis_connected: bool


def load_model() -> Optional[object]:
    """Load the machine learning model from file."""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent.parent / "model" / "text_classifier_model.pkl",
        Path("../model/text_classifier_model.pkl"),
        Path("./model/text_classifier_model.pkl"),
        Path(getattr(settings, "MODEL_PATH", "model/text_classifier_model.pkl")),
    ]

    for model_path in possible_paths:
        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    loaded_model = pickle.load(f)
                logger.info(f"Model loaded successfully from {model_path}")
                return loaded_model
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            continue

    logger.error("Failed to load model from any path")
    return None


async def check_redis_connection() -> bool:
    """Check if Redis connection is working."""
    try:
        redis = Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await redis.ping()
        await redis.close()
        return True
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return False


@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global model

    # Load the ML model
    model = load_model()
    if model is None:
        logger.error("Failed to load model during startup")
        # Don't raise exception here to allow health checks to work

    # Initialize Redis rate limiter
    try:
        redis = Redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await FastAPILimiter.init(redis)
        logger.info("Rate limiter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {e}")
        # Continue without rate limiting for development


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    try:
        await FastAPILimiter.close()
        logger.info("FastAPI limiter closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the index.html file."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/score_text", response_class=HTMLResponse)
async def score_text(request: Request, text: str = Form(...)):
    """Score the text from the form and return an HTML fragment."""
    if model is None:
        return HTMLResponse(
            content="<p>Model not available. Service is temporarily unavailable.</p>",
            status_code=503,
        )

    try:
        # Split text into headline and body
        lines = text.strip().split("\n")
        headline = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

        # Combine for scoring
        text_to_score = f"{headline.strip()} {body.strip()}"

        if len(text_to_score.strip()) < 10:
            return HTMLResponse(
                content="<p>Combined headline and body must be at least 10 characters long.</p>",
                status_code=400,
            )

        # Get prediction
        prediction_proba = model.predict_proba([text_to_score])
        actionable_probability = prediction_proba[0][1]
        intent_score = round(actionable_probability * 100, 1)

        # Determine confidence
        distance_from_neutral = abs(intent_score - 50)
        if distance_from_neutral > 35:
            confidence = "high"
        elif distance_from_neutral > 20:
            confidence = "medium"
        else:
            confidence = "low"

        # Return HTML fragment
        return HTMLResponse(
            content=f"""
                <p><strong>Intent Score:</strong> {intent_score}</p>
                <p><strong>Confidence:</strong> {confidence}</p>
            """,
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return HTMLResponse(
            content="<p>Internal server error during prediction.</p>",
            status_code=500,
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    redis_ok = await check_redis_connection()

    return HealthResponse(
        status="healthy" if (model is not None and redis_ok) else "degraded",
        model_loaded=model is not None,
        redis_connected=redis_ok,
    )


@app.post("/score", response_model=PolicyIntentResponse)
async def score_policy_intent(
    request: PolicyIntentRequest,
    fastapi_request: Request,
    ratelimit: dict = RateLimiter(times=5, seconds=60),  # 5 requests per minute
) -> PolicyIntentResponse:
    """
    Score the policy intent of a press release.

    Returns a score from 0-100 indicating the likelihood that the press release
    represents actionable policy intent rather than political posturing.
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Service is temporarily unavailable.",
        )

    try:
        # Combine headline and body for scoring
        text_to_score = f"{request.headline.strip()} {request.body.strip()}"

        # Basic input validation
        if len(text_to_score.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Combined headline and body must be at least 10 characters long",
            )

        # Get prediction probability
        prediction_proba = model.predict_proba([text_to_score])
        actionable_probability = prediction_proba[0][
            1
        ]  # Probability of "Actionable" class

        # Convert to percentage and round
        intent_score = round(actionable_probability * 100, 1)

        # Determine confidence level
        distance_from_neutral = abs(intent_score - 50)
        if distance_from_neutral > 35:
            confidence = "high"
        elif distance_from_neutral > 20:
            confidence = "medium"
        else:
            confidence = "low"

        logger.info(
            f"Scored text with intent_score: {intent_score}, confidence: {confidence}"
        )

        return PolicyIntentResponse(
            intent_score=intent_score, confidence=confidence, model_version="1.0"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Development server
if __name__ == "__main__":
    import uvicorn

    # Load environment variables for development
    port = int(os.getenv("PORT", 8081))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "score_api:app" if not debug else app,
        host=host,
        port=port,
        reload=debug,
        log_level="info",
    )
