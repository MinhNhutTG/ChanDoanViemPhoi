# 🫁 NGHIÊN CỨU VÀ TRIỂN KHAI MÔ HÌNH CHẨN ĐOÁN VIÊM PHỔI BẰNG TRÍ TUỆ NHÂN TẠO


> Xây dựng hệ thống chẩn đoán viêm phổi từ ảnh X-quang ngực và trực quan hóa kết quả bằng Grad-CAM


![University](https://img.shields.io/badge/Nam%20Can%20Tho%20University-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) 
![AI](https://img.shields.io/badge/Artificial_Intelligence-8A2BE2?style=for-the-badge) 
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) 
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) 
![Groq](https://img.shields.io/badge/Groq-FF4F00?style=for-the-badge&logo=groq&logoColor=white)


## 🙏 Lời cảm ơn

> 💐 Xin chân thành cảm ơn **Thầy Trần Văn Thiện**  
> đã tận tình hướng dẫn, hỗ trợ và đóng góp nhiều ý kiến quý báu  
> trong suốt quá trình thực hiện đề tài.  
>
> 📖 Những kiến thức, kinh nghiệm và sự định hướng của Thầy  
> là nguồn động lực lớn giúp project được hoàn thiện tốt hơn.  
>
> 🤝 Xin chân thành cảm ơn **Nguyễn Gia Bão**  
> đã cùng tham gia thực hiện nghiên cứu, hỗ trợ trong suốt quá trình triển khai project,  
> đóng góp nhiều ý tưởng quan trọng của đề tài.  
>
> 🌟 Sự đồng hành này là một phần quan trọng giúp project đạt được kết quả tốt hơn.
>
> 
> 🌟 **Trân trọng cảm ơn!**

## 🚀 Demo

![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)

Link demo: https://huggingface.co/spaces/MinhNhut3005/chandoanviemphoi

Link demo backup: https://huggingface.co/spaces/MinhNhut3005/ChanDoanViemPhoi_dnc

# 📖 Giới Thiệu Đồ Án



## 🌟 Tính Năng Chính

### 🏥 Phân Tích Ảnh X-quang Ngực
- Tiếp nhận ảnh X-quang ngực (PA/AP).
- Tiền xử lý dữ liệu: resize, normalization.
- Phân loại tình trạng:
  - **Normal**
  - **Pneumonia**

---

## 🤖 Core AI Engine

### 🔍 Phân Loại Ảnh
- Sử dụng mô hình **ResNet50 (CNN)**.
- Huấn luyện trên dữ liệu X-quang ngực.
- Trả về xác suất dự đoán cho từng lớp.

### 🔥 Giải Thích Mô Hình (Explainable AI)
- Ứng dụng kỹ thuật **Grad-CAM**.
- Sinh bản đồ nhiệt (heatmap).
- Overlay heatmap lên ảnh gốc.
- Hiển thị vùng ảnh có ảnh hưởng lớn đến quyết định của mô hình.

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy Dự Án
* **Docker**: Khuyên dùng để chạy bằng container đồng nhất môi trường.
* **Git**: Để clone repository.

## 🏗️ Kiến Trúc Tổng Thể
``` bash
User
  │
  ▼
Frontend (HTML, CSS, JavaScript)
  │
  ▼
Backend API (Flask / FastAPI)
  │
  ▼
Image Processing
  │
  ▼
Deep Learning Model
  │
  ▼
Grad-CAM Visualization
  │
  ▼
Prediction Result
```
## 🖥️ Giao Diện Chính
<img width="1893" height="822" alt="image" src="https://github.com/user-attachments/assets/9f6840ad-2d2f-454a-8e04-c41c068ba6eb" />


### 🛠️ Các Bước Cài Đặt Chi Tiết

#### 1. Clone Repository từ GitHub
Mở terminal và chạy lệnh sau để tải mã nguồn:
```bash
git clone https://github.com/MinhNhutTG/ChanDoanViemPhoi
cd ChanDoanViemPhoi
```
#### 2. Tạo Virtual Environment
```bash
python -m venv .venv
```
#### 3. Kích hoạt môi trường
```bash
Windows: .venv\Scripts\activate
Linux: source .venv/bin/activate
```
#### 4. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```
#### 5. Thiết lập Environment Variables
Tạo file .env trong thư mục gốc với nội dung:
```bash
PORT=your_port
GROQ_API_KEY=your_api_key
```
#### 6. Khởi chạy ứng dụng
▶️ Chạy Backend
```bash
cd backend
python app/app.py
```

#### ✅ Bước 4 — Truy cập frontend
Mở trình duyệt tại: http://localhost


