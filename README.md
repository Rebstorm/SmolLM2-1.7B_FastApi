# SmolLM2-1.7B+FastAPI

SmolLM2-1.7B+FastAPI is a lightweight FastAPI-based wrapper around the `SmolLM2-1.7B-Instruct` model, providing a simple API for text generation with support for streaming and dynamic configuration.

## Quickstart Guide

### Prerequisites
- **Hardware**: At least 8GB of RAM (the model is ~1.7B parameters). A GPU with CUDA support is recommended but not required.
- **Python**: version 3.13 or higher.
- **Tools**: `python` with `pip` (modern standard using `pyproject.toml`), or `docker` for containerized deployment.

### 1. Run with Docker (Recommended)
The easiest way to get started is using Docker.

```bash
# Build the image
docker build -t smol-app .

# Run the container
docker run -p 8000:8000 smol-app
```
The application will be available at `http://localhost:8000`.

### 2. Run Locally
If you prefer to run it outside of Docker:

```bash
# Install dependencies
pipx install .

# Start the application
python main.py
```

### 3. Development Tools
The project uses `ruff` for linting/formatting and `mypy` for type checking.

```bash
# Install dev dependencies
pipx install ".[dev]"

# Run linter
ruff check .

# Run type checker
mypy .
```

## API Usage

### Generate Text
Generate a response for a given prompt.

- **Endpoint**: `GET /generate`
- **Parameters**:
  - `query` (required): The prompt for the model.
  - `stream` (optional): `true` to stream the response (default: `false`).
  - `max_new_tokens` (optional): Maximum number of tokens to generate.

**Example (Standard):**
```bash
curl "http://localhost:8000/generate?query=What+is+2+plus+2"
```

**Example (Streaming):**
```bash
curl "http://localhost:8000/generate?query=Tell+me+a+story&stream=true"
```

### Update Configuration
Modify the system prompt (personality) or default generation limits.

- **Endpoint**: `POST /config`
- **Body**: JSON object with `system_prompt` and/or `max_new_tokens`.

**Example:**
```bash
curl -X POST "http://localhost:8000/config" \
     -H "Content-Type: application/json" \
     -d '{"system_prompt": "You are a pirate.", "max_new_tokens": 50}'
```

## Project Structure
- `app/api/router.py`: API endpoint definitions.
- `app/services/llm_service.py`: LLM logic and model management.
- `app/main.py`: FastAPI application setup and lifespan.
- `main.py`: Entry point script.
