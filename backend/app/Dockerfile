FROM python:3.12-slim

WORKDIR /app

# Cài dependencies hệ thống cho opencv + pydicom
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài torch CPU-only để giảm size (không cần GPU trên Fly.io)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source
COPY . .

EXPOSE 8080

CMD ["python", "app.py"]