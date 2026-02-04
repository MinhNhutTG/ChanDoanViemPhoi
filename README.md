# 🫁 NGHIÊN CỨU VÀ TRIỂN KHAI MÔ HÌNH CHẨN ĐOÁN VIÊM PHỔI BẰNG TRÍ TUỆ NHÂN TẠO


> Ứng dụng hỗ trợ chẩn đoán viêm phổi dựa trên phân tích hình ảnh X-quang phổi và các chỉ số cận lâm sàn bằng Deep Learning


![University](https://img.shields.io/badge/Nam%20Can%20Tho%20University-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) 
![AI](https://img.shields.io/badge/Artificial_Intelligence-8A2BE2?style=for-the-badge) 
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) 
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) 
![Groq](https://img.shields.io/badge/Groq-FF4F00?style=for-the-badge&logo=groq&logoColor=white)


# 📖 Giới Thiệu Đồ Án



## 🌟 Tính Năng Chính

### 🏥 Phân Tích Đa Phương Thức
- **Dữ liệu hình ảnh**: Xử lý ảnh X-quang ngực thẳng (PA/AP).
- **Dữ liệu lâm sàng**: Tích hợp các chỉ số quan trọng như Bạch cầu (WBC), CRP, SpO2, Tuổi, v.v.

### 🤖 Core AI Engine (Backend)
- **Phân Loại (Classification)**: Sử dụng **DenseNet121** để xác định xác suất viêm phổi.
- **Thể hiện vùng ảnh hưởng**: Dùng Grad-CAM để xác định vùng ảnh hướng.
- **Tổng Hợp (Reasoning)**: Sử dụng **LLM Llama 3.3 (via Groq Cloud)** để đóng vai trò bác sĩ, tổng hợp báo cáo.

### 📊 Đánh Giá Rủi Ro Tự Động
- Tự động tính điểm **CURB-65** / **CRB-65** để đánh giá mức độ nghiêm trọng.
- Phân tầng rủi ro (Ngoại trú vs Nhập viện).

## 🚀 Hướng Dẫn Cài Đặt và Chạy Dự Án
* **Docker**: Khuyên dùng để chạy bằng container đồng nhất môi trường.
* **Git**: Để clone repository.



### 🛠️ Các Bước Cài Đặt Chi Tiết

#### 1. Clone Repository từ GitHub
Mở terminal và chạy lệnh sau để tải mã nguồn:
```bash
git clone https://github.com/MinhNhutTG/ChanDoanViemPhoi
cd ChanDoanViemPhoi
```
