import json
from pathlib import Path
import fire
from loguru import logger
from src.training.train import XGBExperiment

def load_config_json(config_json_filename: str):
    """Load a JSON config file
    """
    config_json_filename = Path(config_json_filename)
    if not config_json_filename.exists():
        raise FileNotFoundError(f"Config file {config_json_filename} not found.")
    
    with open(config_json_filename, "r") as f:
        config = json.load(f)
        
    return config


def run_experiment(
        output_dir: str,
        exp_name: str,
        dataset_path: str,
        train_on_cols: list[str],
        target_col: str,
        transformations: dict,
        num_boost_round: int,
        save_pred: bool = False,
        **kwargs_model
    ):
    """Run an XGBoost experiment with the specified parameters."""
    logger.info("Starting experiment with the following parameters:")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Training columns: {train_on_cols}")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Transformations: {transformations}")
    logger.info(f"Number of boosting rounds: {num_boost_round}")
    logger.info(f"Save predictions: {save_pred}")
    logger.info(f"Model parameters: {kwargs_model}")

    experiment = XGBExperiment(
        output_dir=output_dir,
        exp_name=exp_name,
        dataset_path=dataset_path,
        train_on_cols=train_on_cols,
        target_col=target_col,
        transformations=transformations,
        num_boost_round=num_boost_round,
        save_pred=save_pred,
        **kwargs_model
    )
    experiment.run()

def run_experiment_from_config(
        st_config_json_filename: str = None,
        st_checkpoint_dir=None,
        **kwargs
    ):
    """Run an XGBoost experiment with the specified parameters
    
    Args:
        st_config_json_filename: Path to a JSON file containing the experiment configuration. Syne Tune specific.
        **kwargs: Additional keyword arguments to override the configuration file.
    
    """
    config = load_config_json(st_config_json_filename)
    run_experiment(**config, **kwargs)

if __name__ == "__main__":
    fire.Fire(run_experiment_from_config)
