FROM python:3.10-slim

# Set up environment variables to tell Poetry not to create a new virtual environment
# outside the project and to put it in a known location (.venv)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install poetry

RUN poetry install --no-root

COPY . /app

EXPOSE 8000

# Command to run the FastAPI app
CMD ["poetry", "run", "uvicorn",  "api.score_api:app", "--host", "0.0.0.0", "--port", "80"]
