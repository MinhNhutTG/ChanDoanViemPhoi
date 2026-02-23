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
- **Phân Loại (Classification)**: Sử dụng **ResNet** để xác định xác suất viêm phổi.
- **Thể hiện vùng ảnh hưởng**: Dùng Grad-CAM để xác định vùng ảnh hướng.
- **Tổng Hợp (Reasoning)**: Sử dụng **LLM Llama 3.3 (via Groq Cloud)** để đóng vai trò bác sĩ, tổng hợp báo cáo.

### 📊 Đánh Giá Rủi Ro Tự Động
- Tự động tính điểm **CURB-65** / **CRB-65** để đánh giá mức độ nghiêm trọng.
- Phân tầng rủi ro (Ngoại trú vs Nhập viện).

## 🚀 Hướng Dẫn Cài Đặt và Chạy Dự Án
* **Docker**: Khuyên dùng để chạy bằng container đồng nhất môi trường.
* **Git**: Để clone repository.

## 🏗️ Kiến Trúc Tổng Thể
``` bash
[ Web Browser ]
      │
      │ HTTPS POST (clinical data + X-ray image)
      ▼
[ Web Frontend ]
      │
      ▼
[ API Gateway / Backend Server ]
      │
      ├──► [ Clinical Data Processor ]
      │         └─ chuẩn hóa, encode feature vector
      │
      ├──► [ Image Processing Service ]
      │         ├─ tiền xử lý ảnh (resize, normalize)
      │         └─ CNN / Vision Model → image features
      │
      ├──► [ Multimodal Fusion Model ]
      │         ├─ concat / attention fusion
      │         └─ Pneumonia Prediction
      │
      ├──► [ Report Generation Service ]
      │         └─ LLM tạo báo cáo y khoa
      │
      ├──► [ Database / Storage ]
      │         ├─ lưu ảnh X-ray
      │         ├─ lưu thông tin bệnh nhân
      │         └─ lưu kết quả chẩn đoán
      │
      ▼
[ Response Formatter ]
      │
      ▼
[ Web Frontend UI hiển thị kết quả ] 
```
## 🖥️ Giao Diện Chính
<img width="1266" height="777" alt="image" src="https://github.com/user-attachments/assets/0a801fd5-fd4d-414b-8621-9842b39fe28b" />


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
GROQ_API_KEY=your_api_key
```
#### 6. Khởi chạy ứng dụng
▶️ Chạy Backend
```bash
cd backend
python app/main.py
```
▶️ Chạy Frontend
Frontend được phục vụ bằng web server Nginx.

#### ✅ Bước 1 — Tải Nginx

1. Tải Nginx cho Windows từ trang chính thức
2. Giải nén, ví dụ tại:  C:\nginx
3. Sau khi giải nén sẽ có cấu trúc:
   ```bash
   nginx.exe
   conf/
   html/
   logs/
   ```
#### ✅ Bước 2 — Copy mã nguồn frontend

Trong project, mở thư mục:  frontend/
Copy toàn bộ file frontend (HTML, CSS, JS) vào: C:\nginx\html\
Nếu có file mặc định trong thư mục html của Nginx, hãy xóa hoặc ghi đè.

#### ✅ Bước 3 — Khởi động Nginx

Mở PowerShell và chạy:

```powershell
cd C:\nginx
.\nginx.exe
```

#### ✅ Bước 4 — Truy cập frontend
Mở trình duyệt tại: http://localhost

# Thesis
# CHƯƠNG I: ĐẶT VẤN ĐỀ
## 1.1 Tính cấp thiết của đề tài
Viêm phổi là một trong những bệnh lý hô hấp nghiêm trọng, nằm trong nhóm nguyên nhân hàng đầu gây tử vong và nhập viện trên toàn cầu. Tỷ lệ mắc cao cùng diễn tiến bệnh nhanh khiến việc chẩn đoán sớm và chính xác trở thành yếu tố then chốt trong điều trị hiệu quả và giảm nguy cơ biến chứng.

Trong thực hành lâm sàng, chẩn đoán viêm phổi chủ yếu dựa trên hình ảnh X-quang ngực kết hợp với kết quả xét nghiệm cận lâm sàng và thông tin tiền sử bệnh của bệnh nhân. Tuy nhiên, sự gia tăng nhanh chóng về số lượng ca bệnh đã tạo áp lực lớn lên hệ thống y tế, đặc biệt là đội ngũ bác sĩ chẩn đoán hình ảnh. Khối lượng phim cần đọc ngày càng nhiều không chỉ làm tăng nguy cơ quá tải mà còn dẫn đến khả năng chậm trễ trong chẩn đoán và sai sót trong quá trình phân tích hình ảnh, từ đó ảnh hưởng trực tiếp đến chất lượng điều trị và tiên lượng bệnh nhân.

Trong bối cảnh đó, việc ứng dụng các phương pháp hỗ trợ chẩn đoán tự động nhằm nâng cao độ chính xác, rút ngắn thời gian đọc phim và giảm gánh nặng cho nhân viên y tế trở thành một hướng nghiên cứu có ý nghĩa thực tiễn và cấp thiết.

## 1.2 Lý do chọn đề tài
Các nghiên cứu hiện nay trong lĩnh vực trí tuệ nhân tạo y sinh chủ yếu tập trung vào mô hình đơn phương thức, khai thác riêng lẻ dữ liệu hình ảnh y khoa. Tuy nhiên, trong thực tế lâm sàng, quyết định chẩn đoán không chỉ dựa vào hình ảnh mà còn phụ thuộc vào các yếu tố lâm sàng như tuổi, triệu chứng, chỉ số xét nghiệm và tiền sử bệnh.

Việc tích hợp nhiều nguồn dữ liệu khác nhau trong một hệ thống AI đa phương thức cho phép mô hình khai thác thông tin toàn diện hơn, từ đó nâng cao độ chính xác chẩn đoán và khả năng suy luận lâm sàng. Đồng thời, khả năng tự động sinh báo cáo y khoa từ kết quả phân tích sẽ góp phần chuẩn hóa quy trình, tiết kiệm thời gian và hỗ trợ bác sĩ trong thực hành lâm sàng.

Do đó, nghiên cứu xây dựng hệ thống AI đa phương thức phục vụ chẩn đoán viêm phổi là hướng tiếp cận có tính ứng dụng cao và ý nghĩa thực tiễn rõ rệt.

## 1.3 Mục tiêu nghiên cứu
### Mục tiêu tổng quát

Xây dựng hệ thống trí tuệ nhân tạo đa phương thức có khả năng chẩn đoán viêm phổi dựa trên hình ảnh X-quang ngực kết hợp dữ liệu lâm sàng, đồng thời tự động sinh báo cáo y khoa tương ứng.

### Mục tiêu cụ thể

Xây dựng và huấn luyện mô hình học sâu để trích xuất đặc trưng và phân loại bệnh từ hình ảnh X-quang ngực.
Thiết kế cơ chế tích hợp dữ liệu đa phương thức giữa hình ảnh và thông tin lâm sàng.
Phát triển mô-đun sinh báo cáo y khoa tự động dựa trên kỹ thuật xử lý ngôn ngữ tự nhiên.
Đánh giá hiệu năng hệ thống theo các tiêu chí chẩn đoán và chất lượng báo cáo.

## 1.4 Đối tượng và phạm vi
Đối tượng nghiên cứu: Các thuật toán học máy, các bộ dữ liệu y tế về viêm phổi.
Phạm vi ứng dụng: Đề tài tập trung vào việc chẩn đoán viêm phổi thông qua dữ liệu hình ảnh X-quang và dữ liệu lâm sàng.

## 1.5 Ý nghĩa khoa học và thực tiễn
#### Giá trị khoa học:
Đóng góp vào lĩnh vực AI y tế về cách tiếp cận đa phương thức, giúp hiểu rõ hơn sự kết hợp giữa thông tin hình ảnh và ngữ cảnh lâm sàng.
#### Giá trị thực tiễn:
Cung cấp công cụ hỗ trợ bác sĩ ra quyết định nhanh chóng, giảm thiểu sai sót chủ quan và chuẩn hóa quy trình viết báo cáo lâm sàng tại bệnh viện.


# CHƯƠNG II: TỔNG QUAN CƠ SỞ LÝ THUYẾT
## 2.1 Các nghiên cứu tổng quát
## 2.2 Phân tích hệ thống hiện có
## 2.3 Cơ sở lý thuyết
## 2.4 Các công nghệ sử dụng

# CHƯƠNG III: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
## 3.1 Kiến trúc tổng thể
## 3.2 Use Case
## 3.3 Thiết kế dữ liệu
## 3.4 Thiết kế API
## 3.5 Thiết kế giao diện

