FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code
COPY . .

# Cloud Run passes PORT environment variable (defaults to 8080)
EXPOSE 8080

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"

