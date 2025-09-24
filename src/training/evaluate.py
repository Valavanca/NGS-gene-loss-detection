import json
import shap
import numpy as np
import xgboost as xgb
from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calc_optimal_thresholds(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate the optimal thresholds for binary classification using two metrics:
        - Youden's J statistic (Youden Index)
        - Geometric Mean (G-Mean)
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2:
        logger.info("Multiclass classification detected. Returning empty dictionary.")
        return {}

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    metrics = {
        'Youden Index': tpr - fpr,
        'G-Mean': np.sqrt(tpr * (1 - fpr))
    }

    optimal_thresholds = {}
    for metric_name, metric_values in metrics.items():
        idx = np.argmax(metric_values)
        optimal_thresholds[metric_name] = {
            'threshold': float(roc_thresholds[idx]),
            'tpr': float(tpr[idx]),
            'fpr': float(fpr[idx]),
            'value': float(metric_values[idx])
        }

    return optimal_thresholds


def select_top_features(shap_values, top_n=None, frac=None, class_index=None):
    """Select the top features based on SHAP importance values.
    """
    if top_n is not None and frac is not None:
        raise ValueError("Please specify only one of top_n or frac.")
    if top_n is None and frac is None:
        raise ValueError("You must specify either top_n or frac.")
    if frac is not None and not (0 < frac <= 1):
        raise ValueError("Percentage must be between 0 and 1.")
    
    # Multi-class SHAP values
    if len(shap_values.values.shape) == 3:
        # Multi-class: shape (n_samples, n_features, n_classes)
        if class_index is not None:
            shap_vals = shap_values.values[:, :, class_index]
        else:
            # Average across all classes
            shap_vals = np.mean(shap_values.values, axis=2)
    else:
        # Binary or single-class: shape (n_samples, n_features)
        shap_vals = shap_values.values
    
    sorted_features = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=shap_values.feature_names
    ).sort_values(ascending=False)
    sorted_features = sorted_features[sorted_features>0]

    # Select features based on top_n or frac
    if top_n is not None:
        selected_features = sorted_features.head(top_n)
    elif frac is not None:
        num_features = int(len(sorted_features) * frac)
        selected_features = sorted_features.head(num_features)
    
    return selected_features


def evaluate_mclass_model(model, X_test, y_true, output_dir=None):
    """Evaluate the model on the test dataset with multiple metrics
    """
    dtest = xgb.DMatrix(X_test)
    test_predictions_prob = model.predict(dtest)
    test_predictions = np.argmax(test_predictions_prob, axis=1)
    
    accuracy = accuracy_score(y_true, test_predictions)
    logloss = log_loss(y_true, test_predictions_prob)

    precision = precision_score(y_true, test_predictions, average='macro')
    recall = recall_score(y_true, test_predictions, average='macro')
    f1 = f1_score(y_true, test_predictions, average='macro')

    auc = roc_auc_score(y_true, test_predictions_prob, multi_class='ovo')
    conf_matrix = confusion_matrix(y_true, test_predictions)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Logloss: {logloss:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    if output_dir:
        output_dir = Path(output_dir)
        # --- SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        logger.info("Generating SHAP summary plot...")
        with plt.ioff():
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            shap_plot_path = str(output_dir / 'shap_summary_plot.png')
            plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
        logger.info(f"SHAP summary plot saved at: {shap_plot_path}")
    
    metrics = {
        'test_accuracy': accuracy,
        'test_logloss': logloss,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_auc': auc,
        'test_confusion_matrix': conf_matrix
    }
    return metrics

def evaluate_binary_model(model, X_test, y_true, output_dir=None):
    """Evaluate the binary classification model on the test dataset with multiple metrics
    """
    dtest = xgb.DMatrix(X_test)
    test_predictions_prob = model.predict(dtest)  # Probability for 2 classes (binary)
    test_predictions = (test_predictions_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, test_predictions)
    logloss = log_loss(y_true, test_predictions_prob)  # Log-loss for binary classification
    
    precision = precision_score(y_true, test_predictions)
    recall = recall_score(y_true, test_predictions)
    f1 = f1_score(y_true, test_predictions)
    
    # AUC score for binary classification (uses the positive class probability)
    auc = roc_auc_score(y_true, test_predictions_prob)
    conf_matrix = confusion_matrix(y_true, test_predictions)

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Logloss: {logloss:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    if output_dir:
        # --- SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        shap_plot_path = output_dir / 'shap_summary_plot.png'
        with plt.ioff():
            # --- SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            # --- SHAP dependence plot
            shap.dependence_plot('FC PTEN Illumina', shap_values, X_test, show=False)
            plt.savefig(output_dir / 'shap_dependence_plot.png', bbox_inches='tight', dpi=300)
            plt.close()

        logger.info(f"SHAP summary plot saved at: {shap_plot_path}")

        # --- Save predictions to a file
        test_predictions_prob_path = output_dir / 'test_predictions_prob.npy'
        np.save(test_predictions_prob_path, test_predictions_prob)

    metrics = {
        'test_accuracy': accuracy,
        'test_logloss': logloss,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_auc': auc,
        'test_confusion_matrix': conf_matrix
    }
    return metrics