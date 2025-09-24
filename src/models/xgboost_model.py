import xgboost as xgb
from loguru import logger
from config.config import BIOCALL_CONFIG, ZYGOSITY_CONFIG

def create_xgboost_model(target_col: str = 'Bioinf_called_as', custom_params: dict = None) -> xgb.XGBRegressor:
    """Creates an XGBoost model with default or overridden hyperparameters.

    Args:
        custom_params (dict): Optional custom parameters to override defaults.

    Returns:
        xgb.XGBRegressor: An XGBoost regressor instance.
    """
    if custom_params is None:
        custom_params = {}

    if target_col == 'Bioinf called as':
        XGBOOST_PARAMS = BIOCALL_CONFIG
    elif target_col == 'Bioinf Zygosity':
        XGBOOST_PARAMS = ZYGOSITY_CONFIG
    else:
        logger.error(f'Invalid target column: {target_col}')
        return None

    model_params = {**XGBOOST_PARAMS, **custom_params}
    return xgb.XGBRegressor(**model_params)