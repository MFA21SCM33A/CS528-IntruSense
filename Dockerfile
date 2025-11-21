# Base image with Python (Linux; works fine even if you're on Windows host)
FROM python:3.10-slim

# Avoid Python stdout buffering
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Create folders that the code expects
RUN mkdir -p data/raw data/processed models outputs/logs outputs/figures

# Default command: run the stream simulation
# You can override this when running the container
CMD ["python", "src/stream_simulation.py"]
