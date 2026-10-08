import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    DATA_DIR, MODEL_DIR, DISTRICTS, HUONG, PHAP_LY,
    CHAT_LUONG_XAY_DUNG, VIEW_TYPES, WARD_MAPPING,
    TYPE_FEATURES, DU_AN_CAN_HO
)
from train_advanced import encode_data, TYPE_CONFIG
from train_improved import add_features

print("=" * 90)
print("  BANG DANH GIA SAISO MO HINH DU DOAN GIA NHA HCM")
print("=" * 90)

all_results = []

for house_type_name, config in TYPE_CONFIG.items():
    csv_path = config["csv"]
    model_key = config["model_key"]

    if not csv_path.exists():
        print(f"\n[SKIP] {house_type_name}: Khong tim thay {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    if len(df) < 10:
        print(f"\n[SKIP] {house_type_name}: Khong du mau ({len(df)})")
        continue

    model_path = MODEL_DIR / f"{model_key}_model.pkl"
    if not model_path.exists():
        print(f"\n[SKIP] {house_type_name}: Khong tim thay model")
        continue

    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    model_name = model_data.get("model_name", "unknown")

    df_encoded = encode_data(df, house_type_name)
    df_encoded = add_features(df_encoded, house_type_name)
    available_features = [f for f in feature_names if f in df_encoded.columns]
    df_clean = df_encoded[available_features + ["gia"]].dropna()

    X = df_clean[available_features].values
    y_true = df_clean["gia"].values

    feat_df = pd.DataFrame(X, columns=available_features)
    log_pred = model.predict(feat_df)
    y_pred = np.expm1(log_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    median_ae = np.median(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    pct_errors = np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8)) * 100
    under10 = np.mean(pct_errors < 10) * 100
    under20 = np.mean(pct_errors < 20) * 100
    under30 = np.mean(pct_errors < 30) * 100

    bins = [0, 2, 5, 10, 20, 50, 1000]
    labels = ["<2 ty", "2-5 ty", "5-10 ty", "10-20 ty", "20-50 ty", ">50 ty"]
    price_groups = pd.cut(y_true, bins=bins, labels=labels)

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  {house_type_name.upper()} ({model_name})  |  {len(df_clean)} mau  |  {len(available_features)} features")
    print(sep)
    print(f"  MAE:          {mae:.2f} ty VND")
    print(f"  Median AE:    {median_ae:.2f} ty VND")
    print(f"  RMSE:         {rmse:.2f} ty VND")
    print(f"  R2 Score:     {r2:.4f}")
    print(f"  MAPE:         {mape:.1f}%")
    print(f"  Max Error:    {max_error:.2f} ty VND")
    print()
    print(f"  Do chinh xac:")
    print(f"    Sai < 10%:  {under10:.1f}% mau")
    print(f"    Sai < 20%:  {under20:.1f}% mau")
    print(f"    Sai < 30%:  {under30:.1f}% mau")

    print()
    header = f"  {'Muc gia':<12s} {'SL':>5s} {'MAE(ty)':>10s} {'MAPE':>8s} {'RMSE(ty)':>11s}"
    print(header)
    print(f"  {'-'*50}")
    for label in labels:
        mask = price_groups == label
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        mae_g = mean_absolute_error(yt, yp)
        rmse_g = np.sqrt(mean_squared_error(yt, yp))
        mape_g = np.mean(np.abs((yt - yp) / np.maximum(yt, 1e-8))) * 100
        print(f"  {label:<12s} {mask.sum():>5d} {mae_g:>10.2f} {mape_g:>7.1f}% {rmse_g:>11.2f}")

    all_results.append({
        "type": house_type_name,
        "model": model_name,
        "n": len(df_clean),
        "features": len(available_features),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "under10": under10,
        "under20": under20,
    })

print(f"\n{'='*90}")
print(f"  TONG HOP TAT CA MO HINH")
print(f"{'='*90}")
print(f"  {'Loai nha':<16s} {'Model':<12s} {'SL':>4s} {'Feat':>4s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s} {'MAPE':>6s} {'<10%':>6s} {'<20%':>6s}")
print(f"  {'-'*82}")
for r in all_results:
    print(f"  {r['type']:<16s} {r['model']:<12s} {r['n']:>4d} {r['features']:>4d} {r['mae']:>7.2f} {r['rmse']:>7.2f} {r['r2']:>7.4f} {r['mape']:>5.1f}% {r['under10']:>5.1f}% {r['under20']:>5.1f}%")
print(f"  {'-'*82}")
avg_mae = np.mean([r["mae"] for r in all_results])
avg_rmse = np.mean([r["rmse"] for r in all_results])
avg_r2 = np.mean([r["r2"] for r in all_results])
avg_mape = np.mean([r["mape"] for r in all_results])
avg_u10 = np.mean([r["under10"] for r in all_results])
avg_u20 = np.mean([r["under20"] for r in all_results])
print(f"  {'TRUNG BINH':<16s} {'':<12s} {'':>4s} {'':>4s} {avg_mae:>7.2f} {avg_rmse:>7.2f} {avg_r2:>7.4f} {avg_mape:>5.1f}% {avg_u10:>5.1f}% {avg_u20:>5.1f}%")
print()
