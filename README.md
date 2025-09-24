# NGS Gene Loss Detection

A project for predicting gene loss from Next-Generation Sequencing (NGS) data using XGBoost models. The models are trained on TSO500 sequencing data.

## Setup

### Prerequisites
- Python 3.11+ (managed by `uv` if missing)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Valavanca/NGS-gene-loss-detection.git
cd NGS-gene-loss-detection
```

2. Install deps into a local .venv from the lockfile (exact, no drift)
```bash
uv sync --locked
```

3. Use the environment
```bash
uv run --version          # runs with the venv, no manual activation
```

## Models

The project uses two trained XGBoost models for different prediction tasks:
- **Zygosity prediction** (`models/zygosity_p2_733/`)
- **Bioinformatics call** (`models/bioinf_p7_715/`)

### Download Models

Models are stored as project artifacts. Download and extract them to the `models/` folder:
Each model directory should contain:
- `model.json` - Trained XGBoost model
- `info.json` - Model parameters and metrics
- Additional configuration files

## Usage

### Running Inference

The easiest way to run inference is using the following notebook: `./notebooks/inference.ipynb`

### Programmatic Usage

```python
from pathlib import Path
from src.inference.inference import XGBInference

# Zygosity prediction
model = XGBInference(Path('./models/zygosity_p2_733'))
model.run_inference('./data/val_pids_22052025.csv', './output/zyg_pred')

# Bioinformatics prediction  
model = XGBInference(Path('./models/bioinf_p7_715'))
model.run_inference('./data/val_pids_22052025.csv', './output/biocall_pred')
```

## Output

Predictions are saved as JSON files in the `output/` directory:
- `output/zyg_pred/predictions.json` - Zygosity predictions
- `output/biocall_pred/predictions.json` - Bioinformatics predictions