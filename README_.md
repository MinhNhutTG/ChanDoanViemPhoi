# 🫁 Hệ Thống Chẩn Đoán Viêm Phổi (Pneumonia Detection System)

Hệ thống AI hỗ trợ chẩn đoán viêm phổi từ ảnh X-quang ngực (định dạng DICOM), sử dụng mô hình **ResNet50** kết hợp nhiều kỹ thuật học sâu tiên tiến. Backend được xây dựng bằng **FastAPI**, cung cấp API REST cho việc tải ảnh, phân tích và sinh báo cáo lâm sàng tự động.

---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Mô Hình AI](#mô-hình-ai)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Cài Đặt](#cài-đặt)
- [Sử Dụng](#sử-dụng)
- [API Endpoints](#api-endpoints)
- [Notebook Huấn Luyện](#notebook-huấn-luyện)
- [Kỹ Thuật Nâng Cao](#kỹ-thuật-nâng-cao)

---

## Tổng Quan

Hệ thống nhận đầu vào là ảnh X-quang ngực (`.dcm` DICOM hoặc ảnh thông thường `.jpg`, `.png`), phân tích bằng mô hình deep learning và trả về kết quả phân loại kèm độ tin cậy. Ngoài ra, hệ thống có khả năng tạo bản đồ nhiệt **Grad-CAM** để trực quan hóa vùng phổi mà mô hình tập trung vào, cùng với báo cáo lâm sàng tự động thông qua LLM.

**Hai nhãn phân loại:**

| Nhãn | Ý Nghĩa |
|---|---|
| `normal` | Phổi bình thường |
| `lung_opacity` | Phổi có tổn thương / viêm phổi |

---

## Kiến Trúc Hệ Thống

```
Người dùng
    │
    ▼ Upload ảnh X-quang (DICOM / JPG / PNG)
┌─────────────────────────────────────┐
│         FastAPI Backend             │
│  ┌──────────────┐ ┌──────────────┐  │
│  │predict_routes│ │preview_routes│  │
│  └──────┬───────┘ └──────┬───────┘  │
│         │                │          │
│  ┌──────▼───────────────▼───────┐   │
│  │        Services Layer        │   │
│  │  ┌─────────────────────────┐ │   │
│  │  │    predict_service.py   │ │   │
│  │  │  (ResNet50 + TTA Infer) │ │   │
│  │  └─────────────────────────┘ │   │
│  │  ┌─────────────────────────┐ │   │
│  │  │   gradcam_service.py    │ │   │
│  │  │  (Grad-CAM Heatmap)     │ │   │
│  │  └─────────────────────────┘ │   │
│  └───────────────────────────────┘   │
│  ┌────────────────────────────────┐  │
│  │         Utils Layer            │  │
│  │  dicom_util.py │ generate_     │  │
│  │  (DICOM→PIL)   │ report.py     │  │
│  └────────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼ Kết quả JSON (nhãn, confidence, heatmap, báo cáo)
```

---

## Mô Hình AI

### Kiến Trúc ResNet50 Tùy Biến

Backbone **ResNet50** (pretrained ImageNet) được mở rộng với custom head:

```
Input (224×224 RGB)
    │
    ▼
ResNet50 Backbone (frozen ở Phase 1)
    │
    ├──► GAP (Global Average Pooling) → 2048-d
    └──► GMP (Global Max Pooling)     → 2048-d
                    │
                    ▼ Concatenate → 4096-d
              Dropout(0.5)
              Linear(4096 → 512)
              BatchNorm1d
              ReLU
              Dropout(0.25)
              Linear(512 → 2)
                    │
                    ▼
              Logits (Normal / Lung Opacity)
```

### Quá Trình Huấn Luyện 2 Giai Đoạn

| Giai đoạn | Mô tả | Epochs | Learning Rate |
|---|---|---|---|
| **Phase 1** | Freeze backbone, chỉ train custom head | 5 | `1e-3` |
| **Phase 2** | Unfreeze toàn bộ, fine-tune với layered LR | 25 | Backbone: `1e-4`, Head: `5e-4` |

### Thông Số Cấu Hình

| Tham số | Giá trị |
|---|---|
| Input size | 224 × 224 |
| Batch size | 32 |
| Total epochs | 30 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Mixed precision | FP16 (AMP) |

---

## Cấu Trúc Thư Mục

```
ChanDoanViemPhoi/
├── README.md
├── requirements.txt
└── backend/
    ├── models/                         # Lưu model weights (.pth)
    └── app/
        ├── app.py                      # FastAPI app entry point
        ├── routes/
        │   ├── __init__.py
        │   ├── predict_routes.py       # Route chẩn đoán ảnh
        │   └── preview_routes.py       # Route xem trước ảnh DICOM
        ├── services/
        │   ├── __init__.py
        │   ├── predict_service.py      # Logic inference ResNet50 + TTA
        │   └── gradcam_service.py      # Sinh Grad-CAM heatmap
        ├── schemas/
        │   └── response_schema.py      # Pydantic response models
        ├── utils/
        │   ├── __init__.py
        │   ├── dicom_util.py           # Đọc & xử lý file DICOM
        │   └── generate_report.py      # Sinh báo cáo lâm sàng (LLM)
        ├── static/
        │   └── script.js               # Frontend JavaScript
        └── templates/
            └── index.html              # Giao diện web
```

---

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.9+
- CUDA-compatible GPU (khuyến nghị để inference nhanh hơn)

### Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

**Các thư viện chính:**

```
torch torchvision          # Deep learning framework
fastapi uvicorn            # Web framework & server
pydicom                    # Đọc file DICOM
opencv-python-headless     # Xử lý ảnh
Pillow                     # PIL Image
scikit-learn               # Metrics đánh giá
```

### Chuẩn Bị Model

Tải file weights `best_resnet50.pth` vào thư mục `backend/models/`:

```
backend/
└── models/
    └── best_resnet50.pth
```

### Chạy Server

```bash
cd backend
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập giao diện web tại: `http://localhost:8000`

---

## Sử Dụng

### Giao Diện Web

Mở trình duyệt, truy cập `http://localhost:8000`, tải lên ảnh X-quang (DICOM hoặc JPG/PNG) và nhấn **Chẩn đoán**.

### Gọi API trực tiếp

```python
import requests

with open("xray.dcm", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    
result = response.json()
print(result["prediction"])    # "lung_opacity" hoặc "normal"
print(result["confidence"])    # 0.97
```

---

## API Endpoints

### `POST /predict`

Phân tích ảnh X-quang và trả về kết quả chẩn đoán.

**Request:** `multipart/form-data` với trường `file` (`.dcm`, `.jpg`, `.png`)

**Response:**

```json
{
  "prediction": "lung_opacity",
  "confidence": 0.97,
  "probabilities": {
    "normal": 0.03,
    "lung_opacity": 0.97
  },
  "gradcam_image": "<base64_encoded_heatmap>",
  "report": "Kết quả chẩn đoán cho thấy..."
}
```

### `GET /preview`

Xem trước ảnh DICOM đã được chuyển đổi sang PNG để hiển thị trên web.

---

## Notebook Huấn Luyện

File `resnet50-pneumonia.ipynb` (chạy trên Kaggle) ghi lại toàn bộ pipeline huấn luyện:

| Bước | Nội dung |
|---|---|
| 1 | Cài đặt thư viện |
| 2 | Import & cấu hình |
| 3 | Xử lý DICOM → PIL Image (VOI LUT, MONOCHROME normalization) |
| 4 | Data augmentation (train / val / test / TTA) |
| 5 | Dataset & DataLoader với WeightedRandomSampler |
| 6 | Định nghĩa ResNet50 + GAP/GMP custom head |
| 7 | Focal Loss + Label Smoothing |
| 8 | MixUp & CutMix augmentation |
| 9 | Warmup + Cosine Annealing LR scheduler |
| 10 | Training Phase 1 (freeze backbone) |
| 11 | Training Phase 2 (fine-tune toàn bộ) |
| 12 | Đánh giá trên test set (Standard + TTA) |
| 13 | Visualization: Loss / Accuracy / AUC / Confusion Matrix / ROC |
| 14 | Inference đơn ảnh với TTA |

**Dataset:** RSNA Pneumonia Detection Challenge (2-class: `normal` / `lung_opacity`), định dạng DICOM.

---

## Kỹ Thuật Nâng Cao

| Kỹ thuật | Mục đích |
|---|---|
| **Transfer Learning 2 giai đoạn** | Khởi tạo head ổn định trước khi fine-tune toàn bộ |
| **GAP + GMP Concat** | Feature vector phong phú hơn (4096-d thay vì 2048-d) |
| **Focal Loss** | Tập trung học mẫu khó, xử lý class imbalance |
| **Label Smoothing** | Tránh mô hình quá tự tin (overconfident) |
| **WeightedRandomSampler** | Oversample class thiểu số trong mỗi batch |
| **MixUp + CutMix** | Augmentation mạnh bằng cách trộn 2 ảnh |
| **RandomErasing (Cutout)** | Che ngẫu nhiên vùng ảnh, tăng robustness |
| **Warmup + Cosine Annealing** | LR tăng dần rồi giảm mượt, tránh oscillation |
| **Mixed Precision (AMP)** | FP16 training – nhanh hơn, tiết kiệm VRAM |
| **Gradient Clipping** | Tránh gradient exploding, ổn định training |
| **TTA (Test Time Augmentation)** | Average 4 transforms khi inference → kết quả ổn định hơn |
| **Grad-CAM** | Bản đồ nhiệt giải thích vùng phổi mô hình tập trung |
| **Early Stopping** | Dừng sớm khi AUC không cải thiện sau 7 epochs |

---

## 📊 Đánh Giá Mô Hình

Các metric được theo dõi trong quá trình huấn luyện và đánh giá:

- **Accuracy** – Độ chính xác tổng thể
- **AUC-ROC** – Diện tích dưới đường cong ROC (metric chính cho early stopping)
- **F1-Score** – Cân bằng giữa Precision và Recall
- **Confusion Matrix** – Ma trận nhầm lẫn (Standard + TTA)
- **Classification Report** – Precision / Recall / F1 theo từng class

---

## 📝 Ghi Chú

- File model weights `best_resnet50.pth` **không được bao gồm** trong repository do kích thước lớn (~100MB). Cần huấn luyện lại qua notebook hoặc tải về từ Kaggle output.
- Dataset RSNA chứa ảnh DICOM; hệ thống đã xử lý đầy đủ VOI LUT windowing và MONOCHROME1/2 normalization.
- Để chạy inference nhanh nhất, nên dùng GPU với CUDA.
