import os
import sys
import time
import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict

import fire
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from syne_tune import Reporter
from syne_tune.config_space import Domain
import xgboost as xgb


sys.path.append(str(Path(__file__).resolve().absolute().parent.parent))

# -----------------------------------------------------
from src.preprocessing.dataset import load_data, split_data
from src.preprocessing.transforms import prepare_data
from src.training.evaluate import calc_optimal_thresholds, select_top_features, evaluate_binary_model, evaluate_mclass_model
from src.utils import save_json
from config.config import BIN_COLUMNS, SEED, EARLY_STOPPING_ROUNDS

class XGBExperiment:
    def __init__(self, output_dir, exp_name: str, dataset_path, train_on_cols, target_col,
                    transformations, num_boost_round, save_pred=False, **kwargs_model):
        self.exp_name = exp_name
        self.transformations = transformations
        # FIXME: workaround to handle nested search space
        for name, args in transformations.items():
            if isinstance(args, Dict) and len(args) > 0:
                for key, value in args.items():
                    #if isinstance(value, Domain):
                    if key in kwargs_model:
                        transformations[name][key] = kwargs_model.pop(key)

        self.train_on_cols = train_on_cols
        self.target_col = target_col
        self.num_boost_round = num_boost_round
        self.save_pred = save_pred
        self.model_params = {**kwargs_model, 'seed': SEED}
        
        self.scale_pos_weight = None
        self.features_cols = BIN_COLUMNS + ['FC PTEN Illumina', 'TCC GEPADO']

        self.exp_dir = self._create_experiment_dir(output_dir)
        self.dataset_path = str(Path(dataset_path))
        self.data = load_data(dataset_path)
        self.train_ds, self.test_ds = split_data(self.data)

    def _create_experiment_dir(self, output_dir):
        timestamp = time.strftime("%m%d_%H%M_%S")
        rand_suffix = random.randint(1, 9999)
        exp_dir = Path(output_dir) / self.exp_name / f'{timestamp}_{rand_suffix}'
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def _check_label_imbalance(self, labels):
        """Computes and returns a scale_pos_weight for binary classification.

        Args:
            labels (pd.Series): Target labels.

        Returns:
            float: Calculated scale_pos_weight (class ratio), defaults to 1 if not applicable.
        """
        counts = Counter(labels)
        logger.info(f"Class Distribution: {counts}")

        if len(counts) == 2:
            return counts[0] / counts[1] if counts[1] else 1
        return 1
    
    def transform(self, data):
        """Transforms the input data for training and evaluation.
        """
        X, y, _pids = prepare_data(data, self.features_cols, self.target_col, self.transformations) # self.target_col, self.add_poly_degree
        if self.target_col == 'Bioinf called as':
            y = y.map({'WT': 1, 'del': 0})
        elif self.target_col == 'Bioinf Zygosity':
            if not y.isin([0, 1, 2]).all():
                raise ValueError(f"Target column '{self.target_col}' must have only 0, 1, or 2 values.")
        else:
            raise ValueError(f"Target column '{self.target_col}' is not supported.")
        
        return X, y
    
    #@logger.catch
    def run(self):
        """Run the entire experiment pipeline: train, evaluate, and save results."""
        self.train_model()
        self.save()

    def train_model(self):
        """Trains and evaluates an XGBoost model with cross-validation on a provided dataset."""
        logger.info(f"Starting Experiment: {self.exp_name}")
        logger.info(f"Training target '{self.target_col}' using dataset at {str(self.dataset_path)}.")
        logger.info(f"Experiment output will be saved to {self.exp_dir}")

        if len(list(self.exp_dir.glob('*'))) > 0:
            raise ValueError(f"Experiment directory '{self.exp_dir}' already exists and is not empty.")

        X_train, y_train = self.transform(self.train_ds)
        self.scale_pos_weight = self._check_label_imbalance(y_train)

        model, train_info = self.run_xgboost_cv(
            X_train,
            y_train,
            self.model_params,
            self.num_boost_round,
            scale_pos_weight=self.scale_pos_weight
        )
        save_json(train_info, self.exp_dir / 'train_info.json')
        model.save_model(self.exp_dir / "model.ubj")
        cv_metrics = {f'cv_{k}': v for k, v in train_info['cv_best_metrics'].items()}

        save_json(self.test_ds['AN number'].to_list(), self.exp_dir / 'test_AN_numbers.json')
        X_test, y_test = self.transform(self.test_ds)
        test_metrics = self.evaluate(model, X_test, y_test)

        # --- Report metrics to Syne Tune
        report = Reporter()
        combined_metrics = {**cv_metrics, **test_metrics}
        numeric_metrics = {k: v for k, v in combined_metrics.items() if isinstance(v, (int, float))}
        report(**numeric_metrics)
        save_json(combined_metrics, self.exp_dir / 'metrics.json')

    def save(self):
        """Save the experiment configuration and results.
        """
        info = {
            'exp_name': self.exp_name,
            'dataset_path': self.dataset_path,
            'train_on_cols': self.train_on_cols,
            'target_col': self.target_col,
            'transformations': self.transformations,
            'num_boost_round': self.num_boost_round,
            'model_params': self.model_params,
            'scale_pos_weight': self.scale_pos_weight,
            'input_features_cols': self.features_cols,
            'save_pred': self.save_pred,
            'exp_dir': str(self.exp_dir)
        }
        save_json(info, self.exp_dir / 'experiment_info.json')


    def evaluate(self, model, X_test, y_test):
        """Evaluate the trained model on the test dataset.
        """
        dtest = xgb.DMatrix(X_test)
        test_predictions_prob = model.predict(dtest)    

        test_out_dir = self.exp_dir / 'eval'
        test_out_dir.mkdir(parents=True, exist_ok=True)
        # X_test.to_csv(test_out_dir / 'X_test.csv', index=False)
        # y_test.to_csv(test_out_dir / 'y_test.csv', index=False)

        if y_test.nunique() == 2:
            test_metrics = evaluate_binary_model(model, X_test, y_test)
        else:
            test_metrics = evaluate_mclass_model(model, X_test, y_test)
        
        if self.save_pred:
            test_data = {
                'testset_size': X_test.shape,
                'y_distribution': y_test.value_counts().to_dict(),
                'predictions': test_predictions_prob.tolist(),
                'true_labels': y_test.tolist()
            }
            save_json(test_data, test_out_dir / 'test_data.json')

        return test_metrics

    def run_xgboost_cv(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            params: dict,
            num_boost_round: int,
            scale_pos_weight: float = None,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        ) -> tuple:
        """Run XGBoost model with cross-validation.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
            params (dict): XGBoost hyperparameters.
            num_boost_round (int): Number of boosting rounds.
            scale_pos_weight (float): Weight to scale the positive label.
            early_stopping_rounds (int): Early stopping rounds for XGBoost.

        Returns:
            Tuple[xgb.Booster, dict]: Trained model and training details.
        """
        # --- Prepare data
        logger.info(f'Starting Cross-validation: X shape: {X.shape}, y shape: {len(y)}')
        dtrain = xgb.DMatrix(X, label=y)
        if scale_pos_weight and scale_pos_weight > 1:
            params['scale_pos_weight'] = scale_pos_weight

        # pos_y = y.value_counts().get(1, 0)
        pos_y = y.value_counts().min()
        n_folds = min(50, pos_y)
        logger.info(f'Using {n_folds} folds')

        if y.nunique() == 2:
          metrics = {'logloss', 'auc', 'map', 'aucpr'}
          logger.info("Detected binary classification. Metrics: logloss, auc, map, aucpr")
        else:
          metrics = {'mlogloss', 'merror', 'aucpr', 'auc'}

        # --- Cross-validation
        size = dtrain.num_row()
        feature_names = dtrain.feature_names
        
        def fpreproc(dtrain, dtest, param):
            dtrain.feature_names
            dtrain.num_row()
            dtrain.num_col()
            dtrain.get_label()
            return dtrain, dtest, param
        
        best_num_boost_round = params.get('best_num_boost_round', None)
        if not best_num_boost_round:
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                nfold=n_folds,
                # metrics={'logloss', 'auc', 'map', 'aucpr'},
                metrics=metrics,
                early_stopping_rounds=early_stopping_rounds,
                seed=SEED,
                fpreproc=fpreproc,
                verbose_eval=True,
                stratified=True
            )

            optimal_boost_round = len(cv_results) # based on the last metric
            best_metrics = cv_results.iloc[-1]

            logger.info(f"Best number of boosting rounds: {optimal_boost_round}")
            logger.info(f"Best CV metrics: {best_metrics}")
        else:
            optimal_boost_round = int(best_num_boost_round)
            best_metrics = {}

        best_metrics['boost_round'] = optimal_boost_round

        # --- Train the final model
        final_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=optimal_boost_round
        )
        optimal_threshold = calc_optimal_thresholds(y, final_model.predict(xgb.DMatrix(X)))

        # explainer = shap.Explainer(final_model, feature_names=final_model.feature_names)
        # shap_values = explainer(dtrain)
        # feature_importance = select_top_features(shap_values, frac=1)

        train_info = {
            'train_dataset': {
                'n_samples': len(y),
                'features': X.columns.tolist(),
                'y_distribution': y.value_counts().to_dict(),
                'label_imbalance': scale_pos_weight
            },
            'n_folds': n_folds,
            'seed': SEED,
            'num_boost_round': num_boost_round,
            'final_boost_round': optimal_boost_round,
            'cv_best_metrics': best_metrics.to_dict(),
            # 'feature_importance': feature_importance,
            'optimal_threshold': optimal_threshold
        }

        return final_model, train_info
