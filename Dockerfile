FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies directly into the system python
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# main.py calls uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
CMD ["python", "main.py"]
