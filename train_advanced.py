"""
Train 4 model voi cross-validation va model comparison
- RandomForest vs XGBoost vs LightGBM
- 5-fold cross-validation
- Chon model tot nhat cho tung loai nha
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from constants import (
    DATA_DIR, MODEL_DIR, DISTRICTS, HUONG, PHAP_LY,
    VI_TRI_MAT_TIEN, CHAT_LUONG_XAY_DUNG,
    VIEW_TYPES, WARD_MAPPING, TYPE_FEATURES,
    DU_AN_CAN_HO
)

TYPE_CONFIG = {
    "Can ho chung cu": {"csv": DATA_DIR / "can_ho_chung_cu.csv", "model_key": "can_ho"},
}


def encode_data(df, house_type_name):
    df = df.copy()
    df['phuong'] = df.apply(lambda row: WARD_MAPPING.get(f"{row['quan']}_{row['phuong']}", 0), axis=1)
    df['quan'] = df['quan'].map({d: i for i, d in enumerate(DISTRICTS)})
    df['huong_nha'] = df['huong_nha'].map({h: i for i, h in enumerate(HUONG)})
    df['phap_ly'] = df['phap_ly'].map({p: i for i, p in enumerate(PHAP_LY)})

    if house_type_name == "Can ho chung cu":
        df['view'] = df['view'].map({v: i for i, v in enumerate(VIEW_TYPES)})
        df['chat_luong_xay_dung'] = df['chat_luong_xay_dung'].map({c: i for i, c in enumerate(CHAT_LUONG_XAY_DUNG)})
        if 'ten_du_an' in df.columns:
            du_an_map = {name: i for i, name in enumerate(DU_AN_CAN_HO)}
            df['ten_du_an'] = df['ten_du_an'].map(du_an_map).fillna(0).astype(int)

    return df


def get_models():
    """Tra ve dict cac models de so sanh"""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }


def train_and_compare(house_type_name, config):
    """Train va so sanh 3 models voi cross-validation"""
    csv_path = config["csv"]
    model_key = config["model_key"]

    print(f"\n{'='*60}")
    print(f"  {house_type_name.upper()}")
    print(f"{'='*60}")

    if not csv_path.exists():
        print(f"  Khong tim thay file. Bo qua.")
        return None

    df = pd.read_csv(csv_path)
    print(f"So luong mau: {len(df)}")

    if len(df) < 10:
        print(f"  Khong du mau. Bo qua.")
        return None

    df_encoded = encode_data(df, house_type_name)
    feature_names = TYPE_FEATURES[house_type_name]
    available_features = [f for f in feature_names if f in df_encoded.columns]
    df_clean = df_encoded[available_features + ['gia']].dropna()
    print(f"Sau khi loai bo NaN: {len(df_clean)} mau")

    X = df_clean[available_features].values
    y = np.log1p(df_clean['gia'].values)

    models = get_models()
    results = {}

    print(f"\n--- 5-Fold Cross-Validation ---")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        rmse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)

        results[name] = {
            'r2_mean': scores.mean(),
            'r2_std': scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
        }
        print(f"  {name:15s} | R2: {scores.mean():.4f} (+/- {scores.std():.4f}) | RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

    # Chon model tot nhat
    best_name = max(results, key=lambda k: results[k]['r2_mean'])
    best_model = models[best_name]
    print(f"\n  >>> Model tot nhat: {best_name} (R2 = {results[best_name]['r2_mean']:.4f})")

    # Train final model tren toan bo data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    gia_that = np.expm1(y_test)
    gia_pred = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(gia_that, gia_pred))
    mae = mean_absolute_error(gia_that, gia_pred)
    r2 = r2_score(gia_that, gia_pred)

    print(f"\n--- Final Test Set ---")
    print(f"  RMSE: {rmse:.2f} ty")
    print(f"  MAE:  {mae:.2f} ty")
    print(f"  R2:   {r2:.4f}")

    # Feature importance
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)
    print(f"\n--- Feature Importance ({best_name}) ---")
    for fname, imp in feat_imp[:10]:
        print(f"  {fname}: {imp:.4f}")

    # Save model
    model_path = MODEL_DIR / f"{model_key}_model.pkl"
    joblib.dump({
        'model': best_model,
        'feature_names': available_features,
        'house_type': house_type_name,
        'model_name': best_name,
        'cv_r2': results[best_name]['r2_mean'],
    }, model_path)
    print(f"\n  Saved: {model_path}")

    return {
        'type': house_type_name,
        'best_model': best_name,
        'cv_r2': results[best_name]['r2_mean'],
        'test_r2': r2,
        'test_mae': mae,
        'test_rmse': rmse,
        'n_samples': len(df_clean),
        'n_features': len(available_features),
    }


def main():
    print("=" * 60)
    print("  ADVANCED TRAINING - Cross-Validation + Model Comparison")
    print("  RandomForest vs XGBoost vs LightGBM (5-fold CV)")
    print("=" * 60)

    MODEL_DIR.mkdir(exist_ok=True)

    all_results = []
    for house_type_name, config in TYPE_CONFIG.items():
        result = train_and_compare(house_type_name, config)
        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  TONG KET")
    print(f"{'='*60}")
    print(f"{'Loai nha':<20s} {'Model':<15s} {'CV R2':<10s} {'Test R2':<10s} {'MAE (ty)':<10s} {'Samples':<10s}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['type']:<20s} {r['best_model']:<15s} {r['cv_r2']:<10.4f} {r['test_r2']:<10.4f} {r['test_mae']:<10.2f} {r['n_samples']:<10d}")


if __name__ == "__main__":
    main()
