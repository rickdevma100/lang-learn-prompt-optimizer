FROM python:3.12-slim

# Git is required by DVC for experiment tracking
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Copy default prompt files (overridden at runtime by PVC mount)
COPY prompts/ prompts/
COPY metrics/ metrics/

# Copy DVC pipeline config
COPY dvc.yaml params.yaml ./

# Copy entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Initialize Git + DVC so experiments work inside the container
RUN git init && \
    git config user.email "optimizer@lang-learn" && \
    git config user.name "Prompt Optimizer" && \
    dvc init && \
    git add -A && \
    git commit -m "Initial state"

# Expose FastAPI port
EXPOSE 8000

# Start the server via entrypoint (commits runtime state, then starts uvicorn)
ENTRYPOINT ["./entrypoint.sh"]
