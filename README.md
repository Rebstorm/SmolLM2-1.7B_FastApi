# SmolLM2-1.7B+FastAPI

SmolLM2-1.7B+FastAPI is a lightweight FastAPI-based wrapper around the `Qwen2.5-1.5B-Instruct` model, providing a simple API for text generation with support for streaming, dynamic configuration, and retrieval-augmented generation (RAG) over a CV stored in a local Chroma vector database.

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
  - `use_rag` (optional): `true` to ground the response in CV context retrieved from the vector database (default: `false`).
  - `top_k` (optional): Number of CV chunks to retrieve when `use_rag=true` (default: `3`).

**Example (Standard):**
```bash
curl "http://localhost:8000/generate?query=What+is+2+plus+2"
```

**Example (Streaming):**
```bash
curl "http://localhost:8000/generate?query=Tell+me+a+story&stream=true"
```

**Example (RAG over CV):**
```bash
curl "http://localhost:8000/generate?query=What+is+this+person%27s+most+recent+job%3F&use_rag=true"
```

### RAG over CV
On startup, the app looks for a CV PDF and indexes it into a local, persistent Chroma vector database (embeddings via `sentence-transformers/all-MiniLM-L6-v2`), so `/generate?use_rag=true` can retrieve relevant chunks and inject them as context for the model.

- **CV location**: `data/cv.pdf` by default, override with the `CV_PATH` environment variable.
- **Index location**: `data/chroma` by default, override with the `CHROMA_PERSIST_DIR` environment variable.
- The index is only rebuilt when the CV file's contents change (detected via a content hash) — restarting the app with an unchanged CV skips re-embedding.
- If no CV is found at startup, the app still boots; `use_rag=true` requests simply return no retrieved context.

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
- `app/services/rag_service.py`: CV PDF ingestion, chunking, embedding, and Chroma-backed retrieval.
- `app/main.py`: FastAPI application setup and lifespan.
- `main.py`: Entry point script.
- `data/cv.pdf`: CV source document indexed for RAG (not included — add your own).
