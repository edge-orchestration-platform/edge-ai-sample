# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system deps for onnxruntime if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app/
COPY app/requirements.txt ./app/requirements.txt

RUN pip install --upgrade pip && pip install -r ./app/requirements.txt

EXPOSE 8080
CMD ["python", "-m", "app.main"]