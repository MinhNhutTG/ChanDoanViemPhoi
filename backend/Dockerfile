FROM python:3.10-slim

WORKDIR /app

# cài thư viện hệ thống cho opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY backend/app ./app
COPY backend/models ./models

EXPOSE 5000

CMD ["python", "app/app.py"]