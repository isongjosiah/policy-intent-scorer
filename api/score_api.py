from config.config import settings
import pickle
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict
from redis.asyncio import Redis
from fastapi_limiter import FastAPILimiter

# Define the FastAPI application
app = FastAPI()


# Pydantic model for the request body
class PolicyIntentRequest(BaseModel):
    headline: str
    body: str


# Redis and rate limiter setup
# For local development, use a local Redis instance.
# For production, this would point to an Amazon ElastiCache for Redis endpoint.
redis_url = settings.REDIS_URL


@app.on_event("startup")
async def startup():
    redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis)


MODEL_PATH = "model/model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model: pickle = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Ensure it's available.")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")


@app.post("/score")
@FastAPILimiter.limit("5/minute")
async def score_policy_intent(
    request: PolicyIntentRequest, fastapi_request: Request
) -> Dict:
    """
    Accepts a press release and returns a Policy Intent Score.
    """
    try:
        text_to_score = request.headline + " " + request.body
        actionable_probability = model.predict_proba([text_to_score])[0][1]
        intent_score = round(actionable_probability * 100, 1)
        confidence = "high" if abs(intent_score - 50) > 30 else "low"

        return {
            "intent_score": intent_score,
            "confidence": confidence,
            "model_version": "1.0",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
