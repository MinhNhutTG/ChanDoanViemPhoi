# 🫁 NGHIÊN CỨU VÀ TRIỂN KHAI MÔ HÌNH CHẨN ĐOÁN VIÊM PHỔI BẰNG TRÍ TUỆ NHÂN TẠO


> Xây dựng hệ thống chẩn đoán viêm phổi từ ảnh X-quang ngực và trực quan hóa kết quả bằng Grad-CAM


![University](https://img.shields.io/badge/Nam%20Can%20Tho%20University-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) 
![AI](https://img.shields.io/badge/Artificial_Intelligence-8A2BE2?style=for-the-badge) 
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) 
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) 
![Groq](https://img.shields.io/badge/Groq-FF4F00?style=for-the-badge&logo=groq&logoColor=white)


## 🚀 Demo

[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)]
(https://huggingface.co/spaces/MinhNhut3005/chandoanviemphoi)

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
<img width="1827" height="923" alt="image" src="https://github.com/user-attachments/assets/1af55e44-ae3d-4e66-b078-555c8f6846fb" />




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
### 2.1.1 Tổng quan chẩn đoán viêm phổi bằng X-ray
<img width="850" height="318" alt="image" src="https://github.com/user-attachments/assets/72c1a493-6bc8-4b85-8957-79a2012cf964" />

Viêm phổi là một bệnh nhiễm trùng nghiêm trọng ảnh hưởng đến phổi, gây ra bởi vi khuẩn, virus hoặc nấm. Trong y học, chụp X-ray ngực (Chest X-ray) là phương pháp phổ biến để phát hiện các dấu hiệu bất thường trong phổi như:
  * Vùng mờ (opacity)
  * Tổn thương mô phổi
  * Dịch trong phế nang
    
Tuy nhiên, việc đọc ảnh X-ray đòi hỏi bác sĩ chuyên khoa chẩn đoán hình ảnh và dễ xảy ra sai sót khi:

 * Số lượng bệnh nhân lớn
 * Chất lượng ảnh thấp
 * Tổn thương nhỏ khó nhận biết

Vì vậy, nhiều nghiên cứu gần đây đã áp dụng trí tuệ nhân tạo (AI) nhằm hỗ trợ phát hiện bệnh viêm phổi tự động.
### 2.1.2 Ứng dụng Deep Learning trong chẩn đoán hình ảnh y khoa
Deep Learning đã đạt được nhiều thành công trong lĩnh vực Medical Image Analysis. Các mạng nơ-ron tích chập (CNN) có khả năng tự động học đặc trưng từ ảnh y tế.
| Nghiên cứu | Phương pháp | Kết quả |
|:-----|:----:|-----:|
| Rajpurkar et al. | CheXNet (DenseNet121) | Accuracy ~0.76 |
| Kermany et al. | CNN classification | Accuracy ~0.90 |

Các nghiên cứu này cho thấy Deep Learning có thể đạt độ chính xác tương đương hoặc cao hơn bác sĩ trong một số trường hợp.

Tuy nhiên, một hạn chế lớn của các mô hình Deep Learning là tính black-box, nghĩa là khó giải thích tại sao mô hình đưa ra dự đoán.
### 2.1.3 Explainable AI trong y học
Trong các hệ thống AI y tế, việc giải thích kết quả dự đoán là rất quan trọng để:
 * Tăng độ tin cậy của bác sĩ
 * Kiểm chứng tính hợp lý của mô hình
 * Hỗ trợ quá trình chẩn đoán

Explainable AI (XAI) giúp hiển thị vùng ảnh quan trọng mà mô hình sử dụng để đưa ra dự đoán.

Một trong các phương pháp phổ biến nhất là Grad-CAM.
## 2.2 Phân tích hệ thống hiện có
### 2.2.1 Hệ thống chẩn đoán truyền thống
Quy trình chẩn đoán thông thường:
```bash
Bệnh nhân
   ↓
Chụp X-ray
   ↓
Bác sĩ phân tích ảnh
   ↓
Kết luận chẩn đoán
```
Nhược điểm:
 * Phụ thuộc vào kinh nghiệm bác sĩ
 * Mất nhiều thời gian
 * Dễ xảy ra sai sót

### 2.2.2 Hệ thống chẩn đoán dựa trên AI
Các hệ thống AI hiện nay sử dụng pipeline:
```bash
X-ray image
   ↓
Preprocessing
   ↓
Deep Learning Model
   ↓
Prediction
```
Ưu điểm:
 * Xử lý nhanh
 * Tự động hóa
 * Độ chính xác cao
   
Tuy nhiên, nhiều hệ thống vẫn thiếu:
 * Khả năng giải thích kết quả
 * Giao diện thân thiện cho người dùng
## 2.3 Cơ sở lý thuyết
### 2.3.1 Deep Learning và Convolutional Neural Network (CNN)
Deep Learning là một nhánh của Machine Learning sử dụng các mạng nơ-ron sâu để học và trích xuất đặc trưng từ dữ liệu. Trong lĩnh vực xử lý ảnh, Convolutional Neural Network (CNN) là kiến trúc phổ biến và hiệu quả nhất.

CNN là một loại mạng nơ-ron nhân tạo được thiết kế đặc biệt để xử lý dữ liệu hình ảnh. Mạng CNN có khả năng tự động học các đặc trưng quan trọng từ ảnh thay vì phải thiết kế đặc trưng thủ công như các phương pháp truyền thống.

Cấu trúc cơ bản của CNN gồm các lớp chính:
 * Convolution Layer: thực hiện phép tích chập để trích xuất đặc trưng từ ảnh
 * Pooling Layer: giảm kích thước dữ liệu và giữ lại đặc trưng quan trọng
 * Fully Connected Layer: thực hiện quá trình phân loại dựa trên các đặc trưng đã học

```bash
Input Image
   ↓
Convolution
   ↓
Pooling
   ↓
Feature Extraction
   ↓
Fully Connected Layer
   ↓
Classification
```
Nhờ cơ chế học đặc trưng tự động, CNN được ứng dụng rộng rãi trong nhiều bài toán thị giác máy tính như nhận dạng ảnh, phát hiện đối tượng và chẩn đoán hình ảnh y khoa.

### 2.3.2 Transfer Learning
Transfer Learning là phương pháp sử dụng các mô hình đã được huấn luyện trước trên các tập dữ liệu lớn (như ImageNet) để áp dụng cho một bài toán mới. Thay vì huấn luyện mô hình từ đầu, mô hình pretrained có thể được fine-tune để phù hợp với dữ liệu của bài toán.

Một số mô hình CNN phổ biến được sử dụng trong Transfer Learning gồm:
 * ResNet
 * DenseNet
 * EfficientNet

Ưu điểm của Transfer Learning: 
 * Giảm thời gian huấn luyện mô hình
 * Tăng độ chính xác của hệ thống
 * Yêu cầu ít dữ liệu huấn luyện hơn

### 2.3.3 Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) là một phương pháp Explainable AI giúp xác định vùng ảnh quan trọng đối với dự đoán của mô hình.

Nguyên lý:
1. Lấy feature map từ lớp convolution cuối.
2. Tính gradient của lớp dự đoán theo feature map.
3. Tính trọng số trung bình của gradient.
4. Kết hợp các feature map để tạo heatmap.

Công thức: 

$$ LGrad−CAM​=ReLU(k∑​αk​Ak) $$

Trong đó:
 * Ak: Feature map
 * αk: Trọng số gradient

Heatmap được chồng lên ảnh X-ray để hiển thị vùng mà mô hình tập trung.

## 2.4 Các công nghệ sử dụng
Hệ thống trong đề tài sử dụng các công nghệ sau:
### 2.4.1 Python
Python là ngôn ngữ lập trình phổ biến trong lĩnh vực AI và Machine Learning nhờ:
 * Thư viện phong phú
 * Dễ phát triển mô hình
### 2.4.2 PyTorch
PyTorch là framework Deep Learning được sử dụng để:
 * Xây dựng mô hình CNN
 * Huấn luyện mô hình
 * Thực hiện dự đoán
Ưu điểm:
 * Dễ sử dụng
 * Hỗ trợ GPU
 * Cộng đồng lớn
### 2.4.3 OpenCV và thư viện xử lý ảnh
Các thư viện xử lý ảnh được sử dụng để:
 * Đọc ảnh X-ray
 * Resize ảnh
 * Chuẩn hóa dữ liệu
### 2.4.4 Flask / FastAPI
Backend của hệ thống được xây dựng bằng Python framework như:
 * Flask
 * FastAPI
Chức năng:
 * Nhận ảnh từ người dùng
 * Gọi mô hình AI
 * Trả về kết quả dự đoán
### 2.4.5 HTML, CSS và JavaScript
Frontend của website được xây dựng bằng:
 * HTML
 * CSS
 * JavaScript
Chức năng:
 * Upload ảnh X-ray
 * Hiển thị kết quả
 * Hiển thị Grad-CAM heatmap
# CHƯƠNG III: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
## 3.1 Phân tích yêu cầu hệ thống
### 3.1.1 Mục tiêu hệ thống
Hệ thống được xây dựng nhằm hỗ trợ người dùng (bác sĩ hoặc người sử dụng) phát hiện viêm phổi từ ảnh X-ray thông qua mô hình Deep Learning.

Các mục tiêu chính của hệ thống bao gồm:
 * Tự động phân loại ảnh X-ray thành các lớp: Normal, Pneumonia, Not Normal.
 * Hiển thị Grad-CAM heatmap nhằm giải thích vùng ảnh mà mô hình sử dụng để đưa ra dự đoán.
 * Cung cấp giao diện website trực quan để người dùng tải ảnh và xem kết quả chẩn đoán.
### 3.1.2 Yêu cầu chức năng
Các chức năng chính của hệ thống bao gồm:
 * Tải ảnh X-ray: Người dùng có thể tải ảnh X-ray từ máy tính lên hệ thống thông qua giao diện website.
 * Tiền xử lý ảnh: Sau khi ảnh được tải lên, hệ thống sẽ thực hiện các bước tiền xử lý như:
   * Resize ảnh về kích thước phù hợp
   * Chuẩn hóa dữ liệu ảnh
   * Chuyển đổi ảnh sang định dạng phù hợp cho mô hình
 * Dự đoán bệnh: Ảnh sau khi được xử lý sẽ được đưa vào mô hình Deep Learning để thực hiện phân loại.
 * Hiển thị Grad-CAM: Hệ thống sử dụng Grad-CAM để tạo heatmap hiển thị vùng ảnh mà mô hình tập trung khi đưa ra dự đoán.
 * Hiển thị kết quả
   * Kết quả chẩn đoán bao gồm:
   * Nhãn dự đoán
   * Xác suất dự đoán
   * Hình ảnh Grad-CAM

### 3.1.2 Yêu cầu phi chức năng
Ngoài các chức năng chính, hệ thống cần đáp ứng các yêu cầu sau:
* Hiệu suất: Thời gian xử lý và dự đoán nhanh (vài giây).
* Khả năng mở rộng: Có thể tích hợp thêm mô hình AI hoặc dataset mới.
* Tính dễ sử dụng: Giao diện đơn giản, thân thiện với người dùng.
* Độ chính xác: Mô hình Deep Learning phải đạt độ chính xác cao để đảm bảo hỗ trợ chẩn đoán hiệu quả.
## 3.2 Kiến trúc tổng thể
Hệ thống được triển khai theo mô hình Client–Server. Người dùng truy cập hệ thống thông qua trình duyệt web bằng địa chỉ localhost. Máy chủ web Nginx đóng vai trò là reverse proxy, tiếp nhận yêu cầu từ người dùng và chuyển tiếp đến backend API. Backend xử lý yêu cầu và gọi mô hình Deep Learning để thực hiện chẩn đoán ảnh X-ray.

Kiến trúc tổng thể của hệ thống:
```bash
User (Browser)
      │
      ▼
Nginx Web Server
      │
      ▼
Frontend (HTML, CSS, JavaScript)
      │
      ▼
Backend API (Flask / FastAPI)
      │
      ▼
Image Preprocessing
      │
      ▼
Deep Learning Model (ResNet50)
      │
      ▼
Grad-CAM Visualization
      │
      ▼
Prediction Result
```
Các thành phần chính:
Nginx: đóng vai trò web server và reverse proxy, xử lý request từ trình duyệt.
Frontend: Giao diện website, cho phép người dùng upload ảnh X-ray, Hiển thị kết quả dự đoán và Grad-CAM.
Backend: Nhận ảnh từ frontend, xử lý dữ liệu, gọi mô hình AI.
AI Model: Phân loại ảnh X-ray, phát hiện viêm phổi.
Grad-CAM Module: Tạo heatmap giải thích vùng tổn thương.

## 3.3 Use Case
Use Case Diagram mô tả cách người dùng tương tác với hệ thống.

Actor
 * Người dùng (User)
Các Use Case chính:
 * Upload ảnh X-ray
 * Thực hiện chẩn đoán
 * Xem kết quả dự đoán
 * Xem Grad-CAM heatmap
   
![usecase_website_chan_doan_viem_phoi](https://github.com/user-attachments/assets/0f625dc7-7e9a-4976-af78-39db5de26fd4)

Quy trình sử dụng hệ thống:
 * Người dùng truy cập website.
 * Người dùng tải ảnh X-ray lên hệ thống.
 * Hệ thống thực hiện phân tích ảnh.
 * Kết quả dự đoán và Grad-CAM được hiển thị.
## 3.3 Activity Diagram
Activity Diagram mô tả luồng hoạt động của hệ thống khi thực hiện chẩn đoán.
Sơ đồ Activity
Quy trình hoạt động:
1) Người dùng tải ảnh X-ray lên hệ thống.
2) Hệ thống thực hiện tiền xử lý ảnh.
3) Ảnh được đưa vào mô hình Deep Learning.
4) Mô hình trả về kết quả dự đoán.
5) Grad-CAM được tạo để hiển thị vùng ảnh quan trọng.
6) Kết quả được hiển thị trên website.
## 3.4 Sequence Diagram
Sequence Diagram mô tả sự tương tác giữa các thành phần hệ thống theo thời gian.
Sơ đồ: 
## 3.5 Class Diagram
Sơ đồ:
## 3.6 Thiết kế API
Backend của hệ thống cung cấp các API để giao tiếp giữa frontend và mô hình AI.
### 3.6.1 API Upload Image
Endpoint
``` bash
POST /upload
```

Chức năng:
 * Nhận ảnh từ người dùng
 * Lưu ảnh vào server

Request:
```bash
image file
```
Response:
``` bash
{
  "status": "success",
  "image_id": 101
}
```
### 3.6.2 API Predict
Endpoint
``` bash
POST /predict
```

Chức năng:
 * Thực hiện dự đoán ảnh X-ray

Request:
```bash
image file
```
Response:
``` bash
{
 "prediction": "Pneumonia",
 "confidence": 0.91
}
```
## 3.7 Thiết kế giao diện
Giao diện chính: 
<img width="1920" height="963" alt="image" src="https://github.com/user-attachments/assets/11ca8eab-af01-48d5-80cc-19bbef30fdc4" />
Giao diện sau khi upload ảnh
<img width="1913" height="969" alt="image" src="https://github.com/user-attachments/assets/df5f2c00-e9ae-4e7f-9116-8eeb35a6c79f" />
Giao diện kết quả chẩn đoán
<img width="1913" height="969" alt="image" src="https://github.com/user-attachments/assets/54461182-841c-4df2-aefd-14a72851c682" />

## 3.8 Thiết kế mô hình AI


# CHƯƠNG IV: THỬ NGHIỆM VÀ ĐÁNH GIÁ
# CHƯƠNG V: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
# TÀI LIỆU THAM KHẢO


