from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config.config import BIN_COLUMNS, BIN_NORM_REFERENCE

def fit_and_derivative_log_transform(
        row: pd.Series,
        x_cols: list[str],
        degree: int,
        log_base=np.e,
        epsilon=1e-6
    ) -> pd.Series:
    """Fit polynomial, calculate derivative, and apply log transformation
    """
    # Prepare x and y data
    x_numeric = np.arange(len(x_cols))
    y_data = row[x_cols].values.astype(float)

    # --- 1. Fit polynomial of the specified degree
    coeffs = np.polyfit(x_numeric, y_data, degree)
    poly_eqn = np.poly1d(coeffs)
    y_fitted = poly_eqn(x_numeric)
    y_derivative = np.polyder(poly_eqn)(x_numeric)

    # --- 2. Log transformation function
    def log_transform(values, base, epsilon):
        return np.log(np.abs(values) + epsilon) / np.log(base) * np.sign(values)
    
    y_fitted_log = log_transform(y_fitted, log_base, epsilon)
    y_derivative_log = log_transform(y_derivative, log_base, epsilon)
    
    # --- 3. Prepare column names
    fitted_col_names = [f'p{degree}_log_{b}' for b in x_cols]
    derivative_col_names = [f'p{degree}_der_log_{b}' for b in x_cols]
    
    # Return both fitted polynomial values and derivatives as a Series
    result = pd.Series(np.concatenate([y_fitted_log, y_derivative_log]), 
                       index=fitted_col_names + derivative_col_names)
    
    return result

def add_derivative_log_transform(df: pd.DataFrame, degree: int, x_cols: list[str] = BIN_COLUMNS) -> pd.DataFrame:
    """Add polynomial features and derivatives to the dataset.
    """
    poly_bins = df.apply(fit_and_derivative_log_transform, axis=1, x_cols=x_cols, degree=degree)
    return pd.concat([df, poly_bins], axis=1)

def log_transform(df: pd.DataFrame, log_base=np.e, epsilon=1e-6, cols: list[str] = BIN_COLUMNS) -> pd.DataFrame:
    def log_transform_values(values, base, epsilon):
        return np.log(np.abs(values) + epsilon) / np.log(base) * np.sign(values)
    
    df[cols] = df[cols].apply(log_transform_values, base=log_base, epsilon=epsilon)
    return df

def normalize(df: pd.DataFrame, cols: list[str] = BIN_COLUMNS, reference: List = BIN_NORM_REFERENCE) -> pd.DataFrame:
    if len(BIN_NORM_REFERENCE) != len(cols):
        raise ValueError("Length of BIN_NORM_REFERENCE should match the number of columns to normalize.")
    
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in DataFrame.")

    df_norm = df.copy()
    df_norm.loc[:, cols] = df_norm[cols].values / np.array(reference)
    return df_norm

def add_pca(df: pd.DataFrame, n_components: int, cols: list[str] = BIN_COLUMNS) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df[cols])
    pca_df = pd.DataFrame(pca_components, columns=[f'PCA_{i+1}' for i in range(n_components)])
    return pd.concat([df, pca_df], axis=1)

def apply_transformations(df: pd.DataFrame, transformations: dict) -> pd.DataFrame:
    for transform, params in transformations.items():
        if transform == 'poly_deriv_log':
            df = add_derivative_log_transform(df, **params)
        elif transform == 'log_transform':
            df = log_transform(df, **params)
        elif transform == 'normalize':
            df = normalize(df, **params)
        elif transform == 'add_pca':
            df = add_pca(df, **params)
        else:
            raise ValueError(f"Unknown transformation: {transform}")
    return df

def prepare_data(
        data: pd.DataFrame,
        features_cols: list[str],
        target_col: str = None,
        transformations: dict = None
    ) -> tuple:
    """Prepare features and target columns for training/testing.
    """
    if target_col and target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    if any(col not in data.columns for col in features_cols):
        missing_cols = [col for col in features_cols if col not in data.columns]
        raise ValueError(f"Training columns not found in the dataset: {missing_cols}")
    if data.empty:
        raise ValueError("The dataset is empty.")

    if target_col:
        dataset = data.dropna(subset=features_cols + [target_col])
    else:
        dataset = data.dropna(subset=features_cols)

    if dataset.empty:
        raise ValueError("The dataset is empty after dropping rows with missing values.")

    X = dataset[features_cols]
    if transformations:
        X = apply_transformations(X, transformations)

    y = None
    if target_col:
        y = dataset[target_col]
    
    pids = dataset['AN number'].to_list()
    
    return X, y, pids