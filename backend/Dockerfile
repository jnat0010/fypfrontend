# backend/Dockerfile
FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Copy your code
COPY . .

# 3) Expose port & launch
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
