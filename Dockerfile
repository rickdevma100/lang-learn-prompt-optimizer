FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Copy default prompt files (overridden at runtime by PVC mount)
COPY prompts/ prompts/
COPY metrics/ metrics/

# Copy entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Start the server via entrypoint
ENTRYPOINT ["./entrypoint.sh"]
