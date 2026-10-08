"""
Test suite for house price prediction models
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
import json
import tempfile
import joblib

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import (
    DISTRICTS, HUONG, PHAP_LY, CHAT_LUONG_XAY_DUNG,
    VIEW_TYPES, WARD_MAPPING,
    MODEL_DIR, TYPE_FEATURES, DU_AN_CAN_HO, WARDS
)
from train_advanced import encode_data, get_models, TYPE_CONFIG


class TestConstants(unittest.TestCase):
    """Test constants are correctly defined"""

    def test_huong_no_duplicates(self):
        self.assertEqual(len(HUONG), len(set(HUONG)), "HUONG has duplicates")

    def test_phap_ly_no_duplicates(self):
        self.assertEqual(len(PHAP_LY), len(set(PHAP_LY)), "PHAP_LY has duplicates")

    def test_districts_no_duplicates(self):
        self.assertEqual(len(DISTRICTS), len(set(DISTRICTS)), "DISTRICTS has duplicates")

    def test_ward_mapping_consistency(self):
        mapping = WARD_MAPPING
        self.assertIsInstance(mapping, dict)
        self.assertGreater(len(mapping), 0)

    def test_ward_mapping_all_districts_have_wards(self):
        for quan in DISTRICTS:
            self.assertIn(quan, WARDS, f"District {quan} missing from WARDS")

    def test_view_types_no_duplicates(self):
        self.assertEqual(len(VIEW_TYPES), len(set(VIEW_TYPES)))

    def test_du_an_can_ho_count(self):
        self.assertEqual(len(DU_AN_CAN_HO), 17)

    def test_type_features_cover_all_types(self):
        for key in ["Can ho chung cu"]:
            self.assertIn(key, TYPE_FEATURES, f"Missing TYPE_FEATURES for {key}")
            self.assertGreater(len(TYPE_FEATURES[key]), 0)


class TestPreprocessing(unittest.TestCase):
    """Test feature encoding and preprocessing"""

    def test_encode_data_can_ho(self):
        df = pd.DataFrame({
            "dien_tich": [75, 100],
            "quan": ["Quận 7", "Quận 1"],
            "phuong": ["1", "Bến Nghé"],
            "so_phong_ngu": [2, 3],
            "so_phong_tam": [2, 2],
            "tang": [10, 20],
            "tong_so_tang_toa_nha": [25, 30],
            "huong_nha": ["Nam", "Đông"],
            "nam_xay_dung": [2020, 2018],
            "view": ["Thành phố", "Sông"],
            "ten_du_an": ["Vinhomes Grand Park", "Saigon Pearl"],
            "nam_ban_giao": [2024, 2020],
            "phi_quan_ly": [15, 25],
            "co_thang_may": [1, 1],
            "co_ham": [0, 1],
            "phap_ly": ["Sổ hồng", "Sổ hồng"],
            "chat_luong_xay_dung": ["Cao cấp", "Cao cấp"],
            "gia": [3.0, 8.0],
        })
        result = encode_data(df, "Can ho chung cu")
        self.assertIn("ten_du_an", result.columns)
        self.assertTrue(result["ten_du_an"].dtype in [np.int64, np.int32, int])

    def test_get_models_returns_three(self):
        models = get_models()
        self.assertEqual(len(models), 3)
        self.assertIn("RandomForest", models)
        self.assertIn("XGBoost", models)
        self.assertIn("LightGBM", models)


class TestModelFiles(unittest.TestCase):
    """Test saved model files"""

    def test_model_files_exist(self):
        for key in ["can_ho"]:
            model_path = MODEL_DIR / f"{key}_model.pkl"
            if model_path.exists():
                data = joblib.load(model_path)
                self.assertIn("model", data)
                self.assertIn("feature_names", data)
                self.assertIn("house_type", data)

    def test_feature_names_match_type_features(self):
        for key in ["can_ho"]:
            model_path = MODEL_DIR / f"{key}_model.pkl"
            if model_path.exists():
                data = joblib.load(model_path)
                features = data['feature_names']
                self.assertIsInstance(features, list)
                self.assertGreater(len(features), 0)

    def test_model_can_predict(self):
        for key in ["can_ho"]:
            model_path = MODEL_DIR / f"{key}_model.pkl"
            if model_path.exists():
                data = joblib.load(model_path)
                model = data['model']
                features = data['feature_names']
                dummy = pd.DataFrame({f: [0] for f in features})
                pred = model.predict(dummy)
                self.assertEqual(len(pred), 1)
                self.assertFalse(np.isnan(pred[0]))


class TestTypeConfig(unittest.TestCase):
    """Test TYPE_CONFIG consistency"""

    def test_all_types_have_csv_and_model_key(self):
        for name, config in TYPE_CONFIG.items():
            self.assertIn("csv", config, f"{name} missing csv")
            self.assertIn("model_key", config, f"{name} missing model_key")

    def test_model_keys_match_expected(self):
        expected_keys = {"can_ho"}
        actual_keys = {config["model_key"] for config in TYPE_CONFIG.values()}
        self.assertEqual(actual_keys, expected_keys)


if __name__ == '__main__':
    unittest.main()
