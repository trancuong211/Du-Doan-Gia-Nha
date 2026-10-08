"""
Predict Server - Pre-load models khi start, serve predict requests
Chay tren port rieng, Node.js goi HTTP den day
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, request, Response
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)


def jsonify(data):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    return Response(json.dumps(data, default=default), mimetype='application/json')

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
loaded_models = {}

MODEL_KEYS = ["can_ho"]


def train_missing_model(key):
    """Train model thu cong khi file pkl thieu"""
    sys.path.insert(0, str(MODEL_DIR.parent))
    from train_improved import train_and_compare, TYPE_CONFIG
    from constants import MODEL_DIR as _MD
    _MD.mkdir(exist_ok=True)

    TYPE_KEY_MAP = {
        "can_ho": "Can ho chung cu",
    }
    house_type_name = TYPE_KEY_MAP.get(key)
    if not house_type_name or house_type_name not in TYPE_CONFIG:
        logger.error(f"Unknown model key: {key}")
        return False

    logger.info(f"Training missing model: {key} ...")
    result = train_and_compare(house_type_name, TYPE_CONFIG[house_type_name])
    if result:
        logger.info(f"Trained model: {key} (R2={result['test_r2']:.4f})")
        return True
    logger.error(f"Failed to train model: {key}")
    return False


def load_all_models():
    """Load tat ca models khi server start. Auto train neu thieu."""
    MODEL_DIR.mkdir(exist_ok=True)
    start = time.time()
    for key in MODEL_KEYS:
        model_path = MODEL_DIR / f"{key}_model.pkl"
        if not model_path.exists():
            train_missing_model(key)
        if model_path.exists():
            data = joblib.load(model_path)
            loaded_models[key] = data
            logger.info(f"Loaded model: {key} ({data.get('model_name', 'unknown')}, features={len(data['feature_names'])})")
        else:
            logger.warning(f"Model not found after training: {model_path}")
    elapsed = time.time() - start
    logger.info(f"All models loaded in {elapsed:.2f}s")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "model_count": len(loaded_models)
    })


VALID_HOUSE_TYPES = {"can_ho"}

FEATURE_RANGES = {
    "dien_tich": (1, 10000, "float"),
    "so_phong_ngu": (1, 20, "int"),
    "so_phong_tam": (1, 15, "int"),
    "tang": (1, 60, "int"),
    "tong_so_tang_toa_nha": (1, 80, "int"),
    "nam_xay_dung": (1900, 2030, "int"),
    "nam_ban_giao": (1900, 2035, "int"),
    "phi_quan_ly": (0, 500, "float"),
    "quan": (0, 20, "int"),
    "phuong": (0, 500, "int"),
    "huong_nha": (0, 7, "int"),
    "phap_ly": (0, 3, "int"),
    "chat_luong_xay_dung": (0, 2, "int"),
    "view": (0, 3, "int"),
    "co_thang_may": (0, 1, "int"),
    "co_ham": (0, 1, "int"),
    "ten_du_an": (0, 20, "int"),
}


def validate_features(features, feature_names):
    errors = []
    for fname in feature_names:
        val = features.get(fname, 0)
        if fname in FEATURE_RANGES:
            vmin, vmax, ftype = FEATURE_RANGES[fname]
            try:
                if ftype == "int":
                    features[fname] = int(val)
                else:
                    features[fname] = float(val)
            except (TypeError, ValueError):
                errors.append(f"{fname}: invalid type")
                continue
            if features[fname] < vmin or features[fname] > vmax:
                errors.append(f"{fname}: {features[fname]} out of range [{vmin}, {vmax}]")
    return errors


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is empty"}), 400

        house_type = data.get('house_type')
        features = data.get('features', {})

        if not house_type or house_type not in VALID_HOUSE_TYPES:
            return jsonify({"error": f"Invalid house_type: {house_type}. Must be one of {sorted(VALID_HOUSE_TYPES)}"}), 400

        if house_type not in loaded_models:
            return jsonify({"error": f"Model not loaded for house_type: {house_type}"}), 400

        if not isinstance(features, dict):
            return jsonify({"error": "features must be a dict"}), 400

        model_data = loaded_models[house_type]
        model = model_data['model']
        feature_names = model_data['feature_names']

        feat_dict = {f: features.get(f, 0) for f in feature_names}

        validation_errors = validate_features(feat_dict, feature_names)
        if validation_errors:
            return jsonify({"error": "Invalid features", "details": validation_errors}), 400

        feat_df = pd.DataFrame([feat_dict])

        start = time.time()
        log_pred = model.predict(feat_df)[0]
        elapsed = time.time() - start

        prediction = float(np.expm1(log_pred))
        if prediction < 0:
            prediction = 0
        price_vnd = int(prediction * 1_000_000_000)

        logger.info(f"Predict {house_type}: {round(prediction, 2)} ty ({elapsed*1000:.1f}ms)")

        return jsonify({
            "price_billion": round(prediction, 2),
            "price_vnd": price_vnd,
            "model_used": f"{house_type}_model.pkl",
            "house_type": house_type
        })

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_all_models()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    logger.info(f"Starting predict server on port {port}")
    app.run(host='127.0.0.1', port=port, debug=False)
