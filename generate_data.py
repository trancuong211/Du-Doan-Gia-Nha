#!/usr/bin/env python
"""
Generate synthetic data for Can ho chung cu (Condos) to reach 100,000 samples
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path

from constants import (
    DATA_DIR, DISTRICTS, HUONG, PHAP_LY,
    VIEW_TYPES, DU_AN_CAN_HO, WARDS, CHAT_LUONG_XAY_DUNG
)

TARGET_SAMPLES = 100000
CSV_PATH = DATA_DIR / "can_ho_chung_cu.csv"

def load_existing_data():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        return df
    return pd.DataFrame()

def generate_synthetic_data(n_samples):
    np.random.seed(42)
    
    # District probabilities (higher for central districts)
    district_weights = np.array([
        5, 5, 3, 3, 8, 4,   # Q1, Q3, Q4, Q5, Q7, Q10
        7, 6, 5, 5, 8,     # Binh Thanh, Phu Nhuan, Tan Binh, Go Vap, Thu Duc
        6, 5, 4, 3, 2      # Binh Tan, Tan Phu, Q12, Binh Chanh, Nha Be
    ])
    district_weights = district_weights / district_weights.sum()
    
    # Project weights (more popular projects get more samples)
    project_weights = np.ones(len(DU_AN_CAN_HO))
    project_weights[0] = 3  # Vinhomes Grand Park
    project_weights[1] = 3  # Sunrise City
    project_weights[4] = 2  # Saigon Pearl
    project_weights = project_weights / project_weights.sum()
    
    data = []
    for _ in range(n_samples):
        quan = np.random.choice(DISTRICTS, p=district_weights)
        phuong_list = WARDS.get(quan, ["1"])
        phuong = np.random.choice(phuong_list)
        
        # Area: 30-150 m2, skewed towards 50-100
        dien_tich = np.random.lognormal(np.log(70), 0.3)
        dien_tich = np.clip(dien_tich, 30, 150)
        
        # Bedrooms based on area
        if dien_tich < 50:
            so_phong_ngu = np.random.choice([1, 2], p=[0.7, 0.3])
        elif dien_tich < 80:
            so_phong_ngu = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
        else:
            so_phong_ngu = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        
        so_phong_tam = np.random.choice([1, 2], p=[0.4, 0.6]) if so_phong_ngu >= 2 else 1
        
        # Building floors: 15-60
        tong_so_tang_toa_nha = np.random.randint(15, 61)
        tang = np.random.randint(1, tong_so_tang_toa_nha + 1)
        
        # Higher floors slightly more valuable
        floor_premium = 1.0 + (tang / tong_so_tang_toa_nha) * 0.15
        
        huong_nha = np.random.choice(HUONG)
        huong_premium = 1.15 if huong_nha in ["Nam", "Đông Nam", "Đông"] else 1.0
        
        nam_xay_dung = np.random.randint(2000, 2026)
        nam_ban_giao = np.random.randint(nam_xay_dung, min(nam_xay_dung + 5, 2026))
        
        phap_ly = np.random.choice(PHAP_LY, p=[0.4, 0.3, 0.2, 0.1])
        phap_premium = 1.1 if phap_ly in ["Sổ hồng", "Sổ đỏ"] else 0.9
        
        ten_du_an = np.random.choice(DU_AN_CAN_HO, p=project_weights)
        # Premium projects
        project_premium = 1.3 if ten_du_an in ["Vinhomes Grand Park", "Saigon Pearl", "The Sun Avenue"] else \
                         1.15 if ten_du_an in ["Sunrise City", "The Estella", "Masteri Thao Dien", "Vinci Grand Plaza"] else 1.0
        
        view = np.random.choice(VIEW_TYPES, p=[0.1, 0.3, 0.4, 0.2])
        view_premium = 1.2 if view == "Sông" else 1.1 if view == "Thành phố" else 1.0
        
        phi_quan_ly = np.random.uniform(10, 35)
        co_thang_may = np.random.choice([0, 1], p=[0.05, 0.95])
        co_ham = np.random.choice([0, 1], p=[0.7, 0.3])
        chat_luong_xay_dung = np.random.choice(CHAT_LUONG_XAY_DUNG, p=[0.3, 0.5, 0.2])
        quality_premium = 1.3 if chat_luong_xay_dung == "Cao cấp" else 1.0 if chat_luong_xay_dung == "Trung bình" else 0.8
        
        # Base price calculation (ty VND)
        # Base: ~50-80 trieu/m2 depending on location
        district_base = {
            "Quận 1": 120, "Quận 3": 100, "Quận 4": 70, "Quân 5": 65,
            "Quận 7": 85, "Quận 10": 75, "Bình Thạnh": 80, "Phú Nhuận": 95,
            "Tân Bình": 70, "Gò Vấp": 55, "Thủ Đức": 65, "Bình Tân": 50,
            "Tân Phú": 55, "Quận 12": 45, "Bình Chánh": 35, "Nhà Bè": 30
        }.get(quan, 60)
        
        base_price_per_m2 = district_base / 1000  # Convert to ty/m2
        
        # Calculate price
        price = dien_tich * base_price_per_m2
        price *= floor_premium * huong_premium * phap_premium * project_premium * view_premium * quality_premium
        
        # Add noise
        price *= np.random.lognormal(0, 0.08)
        price = np.clip(price, 0.5, 100)  # 0.5 to 100 ty
        
        data.append({
            "dien_tich": round(dien_tich, 1),
            "quan": quan,
            "phuong": phuong,
            "so_phong_ngu": int(so_phong_ngu),
            "so_phong_tam": int(so_phong_tam),
            "huong_nha": huong_nha,
            "nam_xay_dung": int(nam_xay_dung),
            "phap_ly": phap_ly,
            "gia": round(price, 2),
            "tong_so_tang_toa_nha": int(tong_so_tang_toa_nha),
            "tang": int(tang),
            "view": view,
            "ten_du_an": ten_du_an,
            "nam_ban_giao": int(nam_ban_giao),
            "co_thang_may": int(co_thang_may),
            "phi_quan_ly": round(phi_quan_ly, 1),
            "co_ham": int(co_ham),
            "chat_luong_xay_dung": chat_luong_xay_dung,
        })
    
    return pd.DataFrame(data)

def main():
    print(f"Target: {TARGET_SAMPLES:,} samples")
    
    existing_df = load_existing_data()
    n_existing = len(existing_df)
    print(f"Existing samples: {n_existing:,}")
    
    if n_existing >= TARGET_SAMPLES:
        print("Already have enough data!")
        return
    
    n_needed = TARGET_SAMPLES - n_existing
    print(f"Generating {n_needed:,} new samples...")
    
    new_df = generate_synthetic_data(n_needed)
    
    # Combine
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\nSaved to {CSV_PATH}")
    print(f"Total samples: {len(combined_df):,}")
    print(f"Price range: {combined_df['gia'].min():.2f} - {combined_df['gia'].max():.2f} ty")
    print(f"Area range: {combined_df['dien_tich'].min():.1f} - {combined_df['dien_tich'].max():.1f} m2")
    print(f"Districts: {combined_df['quan'].nunique()}")
    print(f"Projects: {combined_df['ten_du_an'].nunique()}")

if __name__ == "__main__":
    main()