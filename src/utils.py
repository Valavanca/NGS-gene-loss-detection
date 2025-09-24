import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def serialize_numpy(obj):
    """Serialization function to handle NumPy arrays and pandas DataFrames/Series."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='split')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.int64, np.float64)):
        return int(obj) if obj % 1 == 0 else float(obj)
    else:
        return str(type(obj))
    

def save_json(data: Any, path: str, indent: int = 4) -> None:
    """Saves data to a JSON file.

    Args:
        data (Any): The data to be saved. Must be JSON-serializable.
        path (str): The file path where the JSON will be saved.
        indent (int, optional): Number of spaces for JSON indentation. Defaults to 4.

    Raises:
        ValueError: If the provided path is invalid.
        TypeError: If the data is not JSON-serializable.
    """
    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with path_obj.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=serialize_numpy)
    except Exception as e:
        raise ValueError(f"Failed to save JSON to {path}: {e}")