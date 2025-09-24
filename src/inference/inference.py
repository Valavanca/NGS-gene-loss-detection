import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from loguru import logger

from src.preprocessing.transforms import prepare_data
from config.config import BIN_COLUMNS
# from config.search_space import TRANSFORMATIONS


from src.utils import save_json

class XGBInference:
    def __init__(self, model_dir, features_cols = BIN_COLUMNS + ['FC PTEN Illumina', 'TCC GEPADO']):
        self.model_dir = Path(model_dir)
        self.features_cols = features_cols
        self.model = self.load_model()
        self.transformations = self.get_transformations()


    def load_model(self):
        """Load the trained XGBoost model from the specified path."""
        model = xgb.Booster()
        if (self.model_dir / 'model.json').exists():
            self.model_path = self.model_dir / 'model.json'
        elif (self.model_dir / 'model.ubj').exists():
            self.model_path = self.model_dir / 'model.ubj'
        else:
            raise FileNotFoundError(f"Model file not found in {self.model_dir}")
        
        model.load_model(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        return model
       
    def get_transformations(self):
        with open(self.model_dir / 'experiment_info.json', 'r') as f:
            trans = json.load(f)['transformations']
        return trans

    def prepare_inference_data(self, data_path):
        """Prepare data for inference.

        Args:
            data_path (str): Path to the dataset for inference.

        Returns:
            pd.DataFrame: Prepared data for inference.
        """
        if isinstance(data_path, pd.DataFrame):
            data = data_path
        else:
            data = pd.read_csv(data_path)
        X_pred, _, pids = prepare_data(data, self.features_cols, target_col=None, transformations=self.transformations)
        return X_pred, pids

    def run_inference(self, data_path, output_dir=None):
        """Run inference on the provided dataset and save the results.

        Args:
            data_path (str): Path to the dataset for inference.
            output_dir (str): Directory to save the inference results.
        """
        X_pred, pids = self.prepare_inference_data(data_path)
        preds = self.model.predict(xgb.DMatrix(X_pred)).tolist()

        pred_results = {
            'ids': pids,
            'preds': preds,
        }
        if output_dir:
            save_json(pred_results, Path(output_dir) / 'predictions.json')
        else:
            return {
                'ids': pids,
                'preds': preds,
                'X_pred': X_pred
            }