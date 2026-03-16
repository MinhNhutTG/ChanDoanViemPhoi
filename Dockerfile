FROM python:3.10-slim

WORKDIR /app

# Cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ backend
COPY backend/ ./backend/

# Port Flask
EXPOSE 8080

CMD ["python", "backend/app/app.py"]