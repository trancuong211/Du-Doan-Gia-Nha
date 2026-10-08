"""
Train Improved - Feature Engineering + Hyperparameter Tuning
Muc tieu: Giam MAPE xuong duoi 10%
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform

from constants import (
    DATA_DIR, MODEL_DIR, DISTRICTS, HUONG, PHAP_LY,
    CHAT_LUONG_XAY_DUNG, VIEW_TYPES, WARD_MAPPING,
    TYPE_FEATURES, DU_AN_CAN_HO
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


def add_features(df, house_type_name):
    """Them features moi tu du lieu hien co"""
    df = df.copy()

    if house_type_name == "Can ho chung cu":
        if 'dien_tich' in df.columns and 'so_phong_ngu' in df.columns:
            df['dien_tich_per_phong'] = df['dien_tich'] / (df['so_phong_ngu'] + 1)
        if 'tang' in df.columns and 'tong_so_tang_toa_nha' in df.columns:
            df['ty_le_tang'] = df['tang'] / (df['tong_so_tang_toa_nha'] + 1)
        if 'phi_quan_ly' in df.columns and 'dien_tich' in df.columns:
            df['phi_x_dien_tich'] = df['phi_quan_ly'] * df['dien_tich']

    return df


def get_param_distributions(model_name):
    """Tham so tuning cho tung model"""
    if model_name == "RandomForest":
        return {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
        }
    elif model_name == "XGBoost":
        return {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2),
        }
    elif model_name == "LightGBM":
        return {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'num_leaves': randint(20, 100),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2),
        }
    return {}


def get_models():
    return {
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    }


def train_and_compare(house_type_name, config):
    csv_path = config["csv"]
    model_key = config["model_key"]

    print(f"\n{'='*70}")
    print(f"  {house_type_name.upper()}")
    print(f"{'='*70}")

    if not csv_path.exists():
        print(f"  Khong tim thay file. Bo qua.")
        return None

    df = pd.read_csv(csv_path)
    print(f"  So luong mau: {len(df)}")

    if len(df) < 10:
        print(f"  Khong du mau. Bo qua.")
        return None

    df_encoded = encode_data(df, house_type_name)
    df_features = add_features(df_encoded, house_type_name)

    feature_names = TYPE_FEATURES[house_type_name]
    new_features = [f for f in df_features.columns if f not in feature_names and f != 'gia' and df_features[f].dtype in ['int64', 'float64', 'int32', 'float32']]
    all_features = feature_names + new_features

    available_features = [f for f in all_features if f in df_features.columns]
    df_clean = df_features[available_features + ['gia']].dropna()
    print(f"  Features: {len(available_features)} (them {len(new_features)} moi)")

    X = df_clean[available_features].values
    y = np.log1p(df_clean['gia'].values)

    models = get_models()
    results = {}

    print(f"\n  --- Baseline (5-fold CV) ---")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        rmse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        results[name] = {
            'r2_mean': scores.mean(),
            'r2_std': scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
        }
        print(f"    {name:12s} | R2: {scores.mean():.4f} | RMSE: {rmse_scores.mean():.4f}")

    # Chon model tot nhat baseline
    best_baseline = max(results, key=lambda k: results[k]['r2_mean'])
    print(f"    >>> Best baseline: {best_baseline}")

    # Hyperparameter tuning
    print(f"\n  --- Hyperparameter Tuning (50 iterations) ---")
    best_score = -999
    best_model = None
    best_name = None

    for name, base_model in models.items():
        param_dist = get_param_distributions(name)
        if not param_dist:
            continue

        search = RandomizedSearchCV(
            base_model, param_dist,
            n_iter=50, cv=5, scoring='r2',
            random_state=42, n_jobs=-1, error_score='raise'
        )
        search.fit(X, y)

        score = search.best_score_
        print(f"    {name:12s} | Best R2: {score:.4f} | Params: {list(search.best_params_.keys())}")

        if score > best_score:
            best_score = score
            best_model = search.best_estimator_
            best_name = name

    print(f"    >>> Best tuned: {best_name} (R2 = {best_score:.4f})")

    # Final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    gia_that = np.expm1(y_test)
    gia_pred = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(gia_that, gia_pred))
    mae = mean_absolute_error(gia_that, gia_pred)
    r2 = r2_score(gia_that, gia_pred)
    mape = np.mean(np.abs((gia_that - gia_pred) / np.maximum(gia_that, 1e-8))) * 100

    print(f"\n  --- Final Test Set ---")
    print(f"    RMSE:  {rmse:.2f} ty")
    print(f"    MAE:   {mae:.2f} ty")
    print(f"    R2:    {r2:.4f}")
    print(f"    MAPE:  {mape:.1f}%")

    under10 = np.mean(np.abs((gia_that - gia_pred) / np.maximum(gia_that, 1e-8)) < 0.1) * 100
    under15 = np.mean(np.abs((gia_that - gia_pred) / np.maximum(gia_that, 1e-8)) < 0.15) * 100
    under20 = np.mean(np.abs((gia_that - gia_pred) / np.maximum(gia_that, 1e-8)) < 0.2) * 100
    print(f"    Sai < 10%: {under10:.1f}%")
    print(f"    Sai < 15%: {under15:.1f}%")
    print(f"    Sai < 20%: {under20:.1f}%")

    # Feature importance
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  --- Top 10 Features ---")
    for fname, imp in feat_imp[:10]:
        print(f"    {fname:35s} {imp:.4f}")

    # Save
    model_path = MODEL_DIR / f"{model_key}_model.pkl"
    joblib.dump({
        'model': best_model,
        'feature_names': available_features,
        'house_type': house_type_name,
        'model_name': best_name,
        'cv_r2': best_score,
        'test_r2': r2,
        'test_mape': mape,
    }, model_path)
    print(f"\n  Saved: {model_path}")

    return {
        'type': house_type_name,
        'best_model': best_name,
        'n_features': len(available_features),
        'cv_r2': best_score,
        'test_r2': r2,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_mape': mape,
        'under10': under10,
        'under15': under15,
        'under20': under20,
        'n_samples': len(df_clean),
    }


def main():
    print("=" * 70)
    print("  IMPROVED TRAINING - Feature Engineering + Hyperparameter Tuning")
    print("=" * 70)

    MODEL_DIR.mkdir(exist_ok=True)

    all_results = []
    for house_type_name, config in TYPE_CONFIG.items():
        result = train_and_compare(house_type_name, config)
        if result:
            all_results.append(result)

    print(f"\n{'='*90}")
    print(f"  TONG KET - SAU KHI CAI THIEN")
    print(f"{'='*90}")
    print(f"  {'Loai nha':<16s} {'Model':<12s} {'Feat':>4s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s} {'MAPE':>6s} {'<10%':>6s} {'<15%':>6s} {'<20%':>6s}")
    print(f"  {'-'*86}")
    for r in all_results:
        print(f"  {r['type']:<16s} {r['best_model']:<12s} {r['n_features']:>4d} {r['test_mae']:>7.2f} {r['test_rmse']:>7.2f} {r['test_r2']:>7.4f} {r['test_mape']:>5.1f}% {r['under10']:>5.1f}% {r['under15']:>5.1f}% {r['under20']:>5.1f}%")
    print(f"  {'-'*86}")

    avg_mae = np.mean([r['test_mae'] for r in all_results])
    avg_rmse = np.mean([r['test_rmse'] for r in all_results])
    avg_r2 = np.mean([r['test_r2'] for r in all_results])
    avg_mape = np.mean([r['test_mape'] for r in all_results])
    avg_u10 = np.mean([r['under10'] for r in all_results])
    avg_u15 = np.mean([r['under15'] for r in all_results])
    avg_u20 = np.mean([r['under20'] for r in all_results])
    print(f"  {'TRUNG BINH':<16s} {'':<12s} {'':>4s} {avg_mae:>7.2f} {avg_rmse:>7.2f} {avg_r2:>7.4f} {avg_mape:>5.1f}% {avg_u10:>5.1f}% {avg_u15:>5.1f}% {avg_u20:>5.1f}%")


if __name__ == "__main__":
    main()
