# backend/Dockerfile
FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy your code
COPY . .

# 3) Expose the port Cloud Run will use and launch with $PORT
#    Cloud Run sets PORT=8080 by default.
EXPOSE 8080
ENV PORT=8080

# Use the shell form so the $PORT env var gets interpolated
CMD uvicorn server:app --host 0.0.0.0 --port $PORT
