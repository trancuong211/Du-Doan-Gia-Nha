# AI Dự Đoán Giá Căn Hộ TP.HCM

Dự án Machine Learning dự đoán giá căn hộ chung cư tại TP.HCM.

## Mô tả

- Dự đoán giá **Căn hộ chung cư** tại 16 quận/huyện TP.HCM
- Model: **XGBoost / LightGBM / RandomForest** với 5-fold Cross-Validation
- Tự động chọn thuật toán tốt nhất qua CV
- Feature engineering: diện tích/phòng, tỷ lệ tầng, phí quản lý × diện tích
- Sai số trung bình (MAPE): **~15-20%** so với giá thực tế

## Hiệu Suất Model

| Loại nhà | Model | R2 | MAE (tỷ) | MAPE | Sai < 10% | Sai < 15% | Sai < 20% |
|----------|-------|-----|----------|------|-----------|-----------|-----------|
| Căn hộ chung cư | XGBoost/LightGBM | ~0.85-0.90 | ~1.5-2.5 | ~15-20% | ~20-30% | ~40-50% | ~55-65% |

*Kết quả có thể khác nhau tùy phiên bản train và dữ liệu.*

## Kiến Trúc Hệ Thống

```
Frontend (HTML/JS)  →  Node.js Server (port 8000)  →  Python Predict Server (port 5001)
                         │                                  │
                         ├── preprocessing.js               ├── Pre-loaded models
                         └── static files                   └── Flask API
```

- **Frontend**: Giao diện nhập thông tin căn hộ (`frontend/`)
- **Node.js Server**: Express server xử lý request, preprocessing, serve frontend (`backend/server.js`)
- **Python Predict Server**: Flask server, pre-load model khi start, serve predictions (`backend/predict_server.py`)

## Cấu Trúc Project

```
Du-Doan-Gia-Nha/
│
├── data/raw/                        # Dữ liệu gốc
│   └── can_ho_chung_cu.csv          # Dữ liệu căn hộ (~200 mẫu)
│
├── models/                          # Model đã train (không commit)
│   └── can_ho_model.pkl
│
├── backend/                         # Web API
│   ├── server.js                    # Express server (port 8000)
│   ├── predict_server.py            # Flask predict server (port 5001)
│   ├── preprocessing.js             # Xử lý input (encode categorical)
│   └── package.json
│
├── frontend/                        # Giao diện web
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── tests/                           # Unit tests
│   └── test_models.py
│
├── reports/                         # Báo cáo, kết quả
│   └── results.md
│
├── constants.py                     # Danh mục dùng chung (quận, phường, hướng, dự án...)
├── train_improved.py                # Train với feature engineering + hyperparameter tuning
├── train_advanced.py                # Train cơ bản (CV + model comparison)
├── predict_cli.py                   # Predict CLI (JSON bridge)
├── evaluate_models.py               # Đánh giá sai số model
├── add_data.py                      # Thêm dữ liệu thủ công
├── generate_data.py                 # Tạo dữ liệu mẫu
├── config.yaml                      # Cấu hình
├── requirements.txt                 # Python dependencies
└── README.md
```

## Features (Căn hộ chung cư - 20 features)

```
dien_tich, quan, phuong, so_phong_ngu, so_phong_tam, tang,
tong_so_tang_toa_nha, huong_nha, nam_xay_dung, view, ten_du_an,
nam_ban_giao, phi_quan_ly, co_thang_may, co_ham, phap_ly,
chat_luong_xay_dung,

# Engineered features:
dien_tich_per_phong, ty_le_tang, phi_x_dien_tich
```

## Cài Đặt

```bash
# Clone repo
git clone <url>
cd Du-Doan-Gia-Nha

# Tạo venv
python -m venv .venv
.venv\Scripts\activate  # Windows

# Cài Python dependencies
pip install -r requirements.txt

# Cài Node.js dependencies
cd backend
npm install
```

> **Lưu ý:** Model files (`.pkl`) không được commit lên git. Khi chạy server lần đầu, predict server sẽ tự động train model nếu chưa có.

## Sử Dụng

### 1. Train model

```bash
# Train với feature engineering + hyperparameter tuning (khuyên dùng)
python train_improved.py

# Train cơ bản (CV + Model Comparison)
python train_advanced.py

# Đánh giá sai số
python evaluate_models.py
```

### 2. Chạy Web App

```bash
# Chạy server (tự start predict server)
cd backend
npm start

# Web app: http://localhost:8000
# Predict server: http://localhost:5001
# Health check: http://localhost:8000/health
```

### 3. Thêm dữ liệu thủ công

```bash
python add_data.py
```

### 4. Predict qua CLI

```bash
# Dùng JSON input
python predict_cli.py can_ho '{"dien_tich": 75, "quan": "Quận 7", "phuong": "Tân Phong", "so_phong_ngu": 2, "so_phong_tam": 2, "tang": 10, "tong_so_tang_toa_nha": 25, "huong_nha": "Nam", "nam_xay_dung": 2020, "view": "Thành phố", "ten_du_an": "Vinhomes Grand Park", "nam_ban_giao": 2024, "phi_quan_ly": 15, "co_thang_may": 1, "co_ham": 1, "phap_ly": "Sổ hồng", "chat_luong_xay_dung": "Cao cấp"}'
```

## Danh Mục Giá Trị

**Quận/Huyện:** Quận 1, Quận 3, Quận 4, Quận 5, Quận 7, Quận 10, Bình Thạnh, Phú Nhuận, Tân Bình, Gò Vấp, Thủ Đức, Bình Tân, Tân Phú, Quận 12, Bình Chánh, Nhà Bè

**Hướng nhà:** Đông, Tây, Nam, Bắc, Đông Nam, Đông Bắc, Tây Nam, Tây Bắc

**Pháp lý:** Sổ hồng, Sổ đỏ, Giấy tờ tay, Đang chờ sổ

**View:** Sông, Thành phố, Nội khu, Không view

**Chất lượng xây dựng:** Cao cấp, Trung bình, Thô

**Tên dự án căn hộ:** Vinhomes Grand Park, Sunrise City, The Estella, Masteri Thao Dien, Saigon Pearl, Vinci Grand Plaza, The Sun Avenue, Icon 56, Saigon Royal, D'Edge, City Garden, Midtown, Charmington, SC VivoCity, Centre Mall, Happy Valley, Millennium

## API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/predict` | POST | Dự đoán giá (body: house_type, features) |
| `/meta` | GET | Lấy danh mục (quận, phường, hướng, dự án...) |
| `/health` | GET | Kiểm tra trạng thái server + predict server |

**Ví dụ request `/predict`:**
```json
{
  "house_type": "can_ho",
  "features": {
    "dien_tich": 75,
    "quan": "Quận 7",
    "phuong": "Tân Phong",
    "so_phong_ngu": 2,
    "so_phong_tam": 2,
    "tang": 10,
    "tong_so_tang_toa_nha": 25,
    "huong_nha": "Nam",
    "nam_xay_dung": 2020,
    "view": "Thành phố",
    "ten_du_an": "Vinhomes Grand Park",
    "nam_ban_giao": 2024,
    "phi_quan_ly": 15,
    "co_thang_may": 1,
    "co_ham": 1,
    "phap_ly": "Sổ hồng",
    "chat_luong_xay_dung": "Cao cấp"
  }
}
```

**Response:**
```json
{
  "price_billion": 4.25,
  "price_vnd": 4250000000,
  "model_used": "can_ho_model.pkl",
  "house_type": "can_ho"
}
```

## License

MIT License