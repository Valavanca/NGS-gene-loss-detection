import os
import pandas as pd
from pathlib import Path
from config.config import DATASET_PATH, TEST_COL

def load_data(file_name: Path = DATASET_PATH) -> pd.DataFrame:
    """Loads data from the processed directory.

    Args:
        file_name (str): Name of the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    file_path = Path(file_name)
    return pd.read_csv(file_path)

def split_data(data: pd.DataFrame, test_column: str = TEST_COL) -> tuple:
    """Splits data into train and test sets.

    Args:
        data (pd.DataFrame): The dataset to split.
        test_column (str): The column indicating test data.

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    test_idx = data[test_column] == 1
    train_idx = data[test_column] == 0

    test_ds = data[test_idx].reset_index(drop=True)
    train_ds = data[train_idx].reset_index(drop=True)

    if train_ds.empty:
        raise ValueError(f"The training dataset is empty. Check the '{test_column}' column values.")
    if test_ds.empty:
        raise ValueError(f"The test dataset is empty. Check the '{test_column}' column values.")
    
    total_records = len(data)
    split_records = len(train_ds) + len(test_ds)
    if total_records != split_records:
        print(f"Warning: Some records may be lost. Total records: {total_records}, Split records: {split_records}")

    return train_ds, test_ds