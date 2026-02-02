# NOTE: intentionally no `# syntax=` directive here to avoid pulling
# `docker.io/docker/dockerfile:*` on restricted networks.

FROM python:3.12-slim

# Set up workdir
WORKDIR /app

# Install system deps (for pandas/pyarrow etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY Agent ./Agent
COPY data ./data

# Default environment: point to Ollama server on host (override as needed)
# Example for Docker Desktop on Windows/Mac: host.docker.internal:11434
# Example for Linux: use host IP or docker network alias
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Default command runs the agent CLI; override args at runtime
# Example: docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 data-agent \
#          python -m Agent.data_agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
ENTRYPOINT ["python", "-m", "Agent.data_agent"]
CMD ["Show me the sales in Nov 2021", "--goal", "Sales trend for Nov 2021"]
