# 🫁 NGHIÊN CỨU VÀ TRIỂN KHAI MÔ HÌNH CHẨN ĐOÁN VIÊM PHỔI BẰNG TRÍ TUỆ NHÂN TẠO


> Xây dựng hệ thống chẩn đoán viêm phổi từ ảnh X-quang ngực và trực quan hóa kết quả bằng Grad-CAM


![University](https://img.shields.io/badge/Nam%20Can%20Tho%20University-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) 
![AI](https://img.shields.io/badge/Artificial_Intelligence-8A2BE2?style=for-the-badge) 
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) 
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) 
![Groq](https://img.shields.io/badge/Groq-FF4F00?style=for-the-badge&logo=groq&logoColor=white)


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
Viêm phổi là một bệnh lý hô hấp nghiêm trọng, thuộc nhóm nguyên nhân hàng đầu gây tử vong và nhập viện trên toàn thế giới. Bệnh có thể tiến triển nhanh, đặc biệt ở trẻ nhỏ, người cao tuổi và bệnh nhân suy giảm miễn dịch. Vì vậy, việc phát hiện sớm và chẩn đoán chính xác đóng vai trò quan trọng trong điều trị kịp thời và giảm thiểu biến chứng.

Trong thực tế lâm sàng, hình ảnh X-quang ngực là phương tiện chẩn đoán phổ biến và có chi phí thấp trong phát hiện viêm phổi. Tuy nhiên, số lượng bệnh nhân ngày càng tăng đã tạo áp lực lớn cho bác sĩ chẩn đoán hình ảnh. Khối lượng phim X-quang cần đọc nhiều có thể dẫn đến quá tải, làm tăng nguy cơ sai sót hoặc chậm trễ trong quá trình đánh giá.

Những năm gần đây, các mô hình học sâu (Deep Learning), đặc biệt là mạng nơ-ron tích chập (CNN), đã chứng minh hiệu quả cao trong phân tích ảnh y tế. Tuy nhiên, phần lớn các mô hình này hoạt động như một “hộp đen”, chỉ đưa ra kết quả dự đoán mà không giải thích được cơ sở của quyết định. Trong lĩnh vực y tế, tính minh bạch và khả năng giải thích là yếu tố rất quan trọng để tạo niềm tin và hỗ trợ bác sĩ trong quá trình ra quyết định.

Do đó, việc xây dựng một hệ thống chẩn đoán viêm phổi từ ảnh X-quang kết hợp với cơ chế giải thích bằng kỹ thuật Grad-CAM nhằm hiển thị vùng ảnh quan trọng là một hướng nghiên cứu có ý nghĩa thực tiễn cao.

## 1.2 Lý do chọn đề tài
Mặc dù các mô hình học sâu có thể đạt độ chính xác cao trong phân loại viêm phổi, nhưng nếu chỉ đưa ra nhãn dự đoán (ví dụ: “Pneumonia” hoặc “Normal”) thì chưa đủ để ứng dụng trong môi trường lâm sàng. Bác sĩ cần biết mô hình dựa vào vùng nào trên ảnh để đưa ra kết luận.

Kỹ thuật Grad-CAM (Gradient-weighted Class Activation Mapping) cho phép trực quan hóa vùng ảnh có ảnh hưởng lớn nhất đến quyết định của mô hình thông qua bản đồ nhiệt (heatmap). Việc hiển thị vùng phổi được mô hình “chú ý” sẽ giúp:

 * Tăng tính minh bạch của hệ thống AI

 * Hỗ trợ bác sĩ kiểm tra lại kết quả dự đoán

 * Phát hiện các trường hợp mô hình học sai hoặc tập trung vào vùng không phù hợp

Vì vậy, đề tài tập trung xây dựng hệ thống chẩn đoán viêm phổi dựa trên học sâu và tích hợp cơ chế giải thích Grad-CAM để nâng cao tính tin cậy và khả năng ứng dụng thực tế.

## 1.3 Mục tiêu nghiên cứu
### Mục tiêu tổng quát

 * Xây dựng hệ thống học sâu có khả năng chẩn đoán viêm phổi từ ảnh X-quang ngực và trực quan hóa vùng ảnh quan trọng thông qua kỹ thuật Grad-CAM.

### Mục tiêu cụ thể

 * Thu thập và tiền xử lý dữ liệu ảnh X-quang ngực phục vụ huấn luyện mô hình.

 * Xây dựng và huấn luyện mô hình CNN (ví dụ: ResNet) để phân loại viêm phổi.

 * Tích hợp phương pháp Grad-CAM để sinh bản đồ nhiệt thể hiện vùng ảnh ảnh hưởng đến dự đoán.

 * Đánh giá hiệu năng mô hình theo các chỉ số như Accuracy, Precision, Recall, F1-score và Confusion Matrix.

 * Phân tích và so sánh giữa kết quả dự đoán và vùng được mô hình chú ý nhằm đánh giá tính hợp lý của hệ thống.

## 1.4 Đối tượng và phạm vi
### Đối tượng nghiên cứu:
 * Các mô hình học sâu dùng trong phân tích ảnh y tế.

 * Kỹ thuật giải thích mô hình (Explainable AI), cụ thể là Grad-CAM.

 * Dữ liệu ảnh X-quang ngực phục vụ chẩn đoán viêm phổi.
### Phạm vi nghiên cứu:
 * Đề tài chỉ tập trung vào dữ liệu hình ảnh X-quang ngực.

 * Hệ thống thực hiện phân loại viêm phổi (2 hoặc 3 lớp tùy bộ dữ liệu).

 * Không bao gồm tích hợp dữ liệu lâm sàng hay sinh báo cáo tự động.

## 1.5 Ý nghĩa khoa học và thực tiễn
#### Giá trị khoa học:
 * Góp phần nghiên cứu và ứng dụng Explainable AI trong lĩnh vực chẩn đoán hình ảnh y tế.

 * Làm rõ vai trò của Grad-CAM trong việc tăng tính minh bạch và khả năng giải thích của mô hình học sâu.
#### Giá trị thực tiễn:
 * Hỗ trợ bác sĩ trong quá trình đọc phim X-quang bằng cách cung cấp thêm thông tin trực quan về vùng nghi ngờ viêm phổi.

 * Giảm áp lực phân tích thủ công khi số lượng bệnh nhân lớn.

 * Tăng mức độ tin cậy khi ứng dụng AI trong môi trường lâm sàng.


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

