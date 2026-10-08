"""
Script them du lieu nha moi vao dataset
Moi loai nha co 1 file CSV rieng

Cach dung: py add_data.py
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from pathlib import Path
import subprocess

from constants import (
    DISTRICTS, HUONG, PHAP_LY,
    VI_TRI_MAT_TIEN, CHAT_LUONG_XAY_DUNG,
    VIEW_TYPES, DU_AN_CAN_HO, WARDS
)

DATA_DIR = Path("data/raw")

CSV_FILES = {
    "Can ho chung cu": DATA_DIR / "can_ho_chung_cu.csv",
}

HOUSE_TYPES = ["Can ho chung cu"]

CAN_HO_COLUMNS = [
    "dien_tich", "quan", "phuong", "so_phong_ngu", "so_phong_tam",
    "tang", "tong_so_tang_toa_nha", "huong_nha", "nam_xay_dung",
    "view", "ten_du_an", "nam_ban_giao", "phi_quan_ly",
    "co_thang_may", "co_ham", "phap_ly", "chat_luong_xay_dung", "gia"
]

TYPE_COLUMNS = {
    "Can ho chung cu": CAN_HO_COLUMNS,
}


def prompt_choice(label, options):
    print(f"\n{label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input("Chon so: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Vui long chon so hop le.")


def prompt_number(label, cast=float, default=None):
    hint = f" [mac dinh: {default}]" if default is not None else ""
    while True:
        value = input(f"{label}{hint}: ").strip()
        if value == "" and default is not None:
            return default
        try:
            return cast(value)
        except ValueError:
            print("Vui long nhap so hop le.")


def main():
    print("=== THEM DU LIEU NHA MOI ===")

    house_type = "Can ho chung cu"

    dien_tich = prompt_number("Dien tich (m2)")
    so_phong_ngu = prompt_number("So phong ngu", int)
    so_phong_tam = prompt_number("So phong tam", int)
    so_tang = prompt_number("So tang", int)
    quan = prompt_choice("Quan/Huyen", DISTRICTS)
    phuong = input("Phuong: ").strip() or "1"
    huong_nha = prompt_choice("Huong nha", HUONG)
    nam_xay_dung = prompt_number("Nam xay dung", int, default=2020)
    phap_ly = prompt_choice("Phap ly", PHAP_LY)
    gia = prompt_number("Gia (ty VND)")

    new_row = {
        "dien_tich": dien_tich, "quan": quan, "phuong": phuong,
        "so_phong_ngu": so_phong_ngu, "so_phong_tam": so_phong_tam,
        "so_tang": so_tang, "huong_nha": huong_nha,
        "nam_xay_dung": nam_xay_dung,
        "khoang_cach_trung_tam": prompt_number("Khoang cach den trung tam (km)", default=5),
        "phap_ly": phap_ly, "gia": gia,
    }

    new_row["tang"] = prompt_number("Tang hien tai", int, default=10)
    new_row["tong_so_tang_toa_nha"] = prompt_number("Tong so tang toa nha", int, default=25)
    new_row["view"] = prompt_choice("View", VIEW_TYPES)
    print("\nDu an:")
    for i, d in enumerate(DU_AN_CAN_HO, 1):
        print(f"  {i}. {d}")
    new_row["ten_du_an"] = prompt_choice("Du an", DU_AN_CAN_HO)
    new_row["nam_ban_giao"] = prompt_number("Nam ban giao", int, default=2024)
    new_row["phi_quan_ly"] = prompt_number("Phi quan ly (nghin/m2/thang)", default=15)
    new_row["co_thang_may"] = 1 if input("Co thang may? (y/n): ").strip().lower() == "y" else 1
    new_row["co_ham"] = 1 if input("Co ham? (y/n): ").strip().lower() == "y" else 0
    new_row["chat_luong_xay_dung"] = prompt_choice("Chat luong xay dung", CHAT_LUONG_XAY_DUNG)
    new_row["mat_tien"] = 0

    csv_path = CSV_FILES[house_type]
    columns = TYPE_COLUMNS[house_type]

    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=columns)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.reindex(columns=columns)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nDa them vao {csv_path}. Tong so dong: {len(df)}")

    retrain = input("Train lai model ngay? (y/n): ").strip().lower()
    if retrain == "y":
        subprocess.run(["python", "train_advanced.py"])
    else:
        print("Bo qua train lai. Chay 'python train_advanced.py' bat cu luc nao.")


if __name__ == "__main__":
    main()
