"""
Constants dung chung cho toan bo project
- train_advanced.py
- predict_cli.py
- add_data.py
"""
from pathlib import Path

# Data paths
DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

# Districts
DISTRICTS = [
    "Quận 1", "Quận 3", "Quận 4", "Quận 5", "Quận 7", "Quận 10",
    "Bình Thạnh", "Phú Nhuận", "Tân Bình", "Gò Vấp", "Thủ Đức",
    "Bình Tân", "Tân Phú", "Quận 12", "Bình Chánh", "Nhà Bè"
]

# House types
HOUSE_TYPES = ["Căn hộ chung cư"]
TYPE_KEYS = ["can_ho"]

# Categorical features
HUONG = ["Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"]
PHAP_LY = ["Sổ hồng", "Sổ đỏ", "Giấy tờ tay", "Đang chờ sổ"]
VI_TRI_MAT_TIEN = ["Mặt tiền đường lớn", "Mặt tiền hẻm", "Mặt tiền ngã tư"]
CHAT_LUONG_XAY_DUNG = ["Cao cấp", "Trung bình", "Thô"]
LOAI_BIET_THU = ["Đơn lập", "Song lập", "Hàng kề"]
VIEW_TYPES = ["Sông", "Thành phố", "Nội khu", "Không view"]
VI_TRI_HEM = ["Hẻm thông", "Hẻm cụt"]
DU_AN_CAN_HO = [
    "Vinhomes Grand Park", "Sunrise City", "The Estella", "Masteri Thao Dien",
    "Saigon Pearl", "Vinci Grand Plaza", "The Sun Avenue", "Icon 56",
    "Saigon Royal", "D'Edge", "City Garden", "Midtown", "Charmington",
    "SC VivoCity", "Centre Mall", "Happy Valley", "Millennium"
]

# Ward mapping: {quan: [phuong]}
WARDS = {
    "Quận 1": ["Bến Nghé", "Bến Thành", "Cầu Kho", "Cầu Ông Lãnh", "Đa Kao", "Nguyễn Cư Trinh", "Nguyễn Thái Bình", "Tân Định"],
    "Quận 3": ["Phú Nhuận", "Võ Thị Sáu", "9", "10", "11", "12", "13", "14"],
    "Quận 4": ["Tân Thuận Đông", "Tân Thuận Tây", "Tân Kiểng", "Tân Hưng", "Bình Thuận", "Phú Thuận"],
    "Quận 5": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    "Quận 7": ["Tân Phong", "Tân Phú", "Tân Kiểng", "Tân Hưng", "Bình Thuận", "Phú Thuận", "Tân Quy", "Tân Mỹ"],
    "Quận 10": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    "Bình Thạnh": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
    "Phú Nhuận": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"],
    "Tân Bình": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    "Gò Vấp": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"],
    "Thủ Đức": ["Linh Chiểu", "Linh Tây", "Linh Đông", "Linh Trung", "Bình Chiểu", "Hiệp Bình Chánh", "Hiệp Bình Phước", "Tam Phú", "Trường Thọ"],
    "Bình Tân": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    "Tân Phú": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
    "Quận 12": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    "Bình Chánh": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
    "Nhà Bè": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
}


def build_ward_mapping():
    """Tao dict map {quan_phuong: index} de encode phuong khong trung lap"""
    mapping = {}
    idx = 0
    for quan, phuongs in WARDS.items():
        for phuong in phuongs:
            key = f"{quan}_{phuong}"
            if key not in mapping:
                mapping[key] = idx
                idx += 1
    return mapping


WARD_MAPPING = build_ward_mapping()


# Features per house type
COMMON_FEATURES = [
    "dien_tich", "quan", "phuong", "so_phong_ngu", "so_phong_tam", "so_tang",
    "huong_nha", "nam_xay_dung", "mat_tien",
    "khoang_cach_trung_tam", "phap_ly"
]

TYPE_FEATURES = {
    "Can ho chung cu": [
        "dien_tich", "quan", "phuong", "so_phong_ngu", "so_phong_tam",
        "tang", "tong_so_tang_toa_nha", "huong_nha", "nam_xay_dung",
        "view", "ten_du_an", "nam_ban_giao", "phi_quan_ly",
        "co_thang_may", "co_ham", "phap_ly", "chat_luong_xay_dung"
    ],
}

# CSV columns per type
CAN_HO_EXTRA = ["tong_so_tang_toa_nha", "tang", "view", "ten_du_an", "nam_ban_giao",
                "co_thang_may", "phi_quan_ly", "co_ham", "chat_luong_xay_dung"]

CSV_COLUMNS = {
    "Can ho chung cu": [c for c in COMMON_FEATURES if c not in ["so_tang", "mat_tien", "khoang_cach_trung_tam"]] + ["gia"] + CAN_HO_EXTRA,
}

# Model paths
MODEL_MAP = {
    "can_ho": MODEL_DIR / "can_ho_model.pkl",
}
