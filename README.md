# NGS PTEN Gene Loss Detection

Code for detecting PTEN deletions in TSO500 (TruSight Oncology 500) panel sequencing data with XGBoost.

The standard TSO500 pipeline averages bin counts over the whole gene, so small focal deletions in PTEN often go unnoticed. Here we train XGBoost models on the per-bin normalized counts across the PTEN locus instead. The models predict whether a deletion is present and what its zygosity is.

This repository accompanies the paper [Machine learning uncovers previously undetectable genetic deletions in cancer panel sequencing](https://www.nature.com/articles/s41698-026-01756-9) (npj Precision Oncology, 2026).

## Setup

### Prerequisites
- Python 3.11+ (`uv` will install it if you don't have it)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Valavanca/NGS-gene-loss-detection.git
cd NGS-gene-loss-detection
```

2. Install the dependencies into a local `.venv` from the lockfile (exact versions, no drift):
```bash
uv sync --locked
```

3. Run things inside that environment with `uv run`, no manual activation needed. Jupyter isn't a project dependency, so for the notebook pull it in on the fly:
```bash
uv run --with jupyter jupyter lab
```

## Models

Two trained models are included:
- **Bioinformatics call** (`models/bioinf_p7_715/`): a binary classifier for PTEN status. It was trained with `WT = 1` and `del = 0`, so the output is the probability that the sample is wild type. Low values point to a deletion.
- **Zygosity** (`models/zygosity_p2_733/`): a 3-class classifier (classes 0, 1, 2) that outputs one probability per class.

Each model directory contains:
- `model.json`: the trained XGBoost booster
- `experiment_info.json`: the feature transformation applied before prediction (inference reads it, so keep it next to the model)
- `info.json`: hyperparameters plus cross-validation and held-out test metrics
- `train_info.json`, `cv_results.json`: details from the training run

## Input data

The patient data used in the paper is not part of this repository. To run the models on your own samples, prepare a CSV with one row per sample and these columns:
- `AN number`: sample ID, copied into the output
- `chr10.136_NC` … `chr10.174_NC`: the 39 normalized bin counts covering PTEN (the full list is `BIN_COLUMNS` in `config/config.py`)
- `FC PTEN Illumina`: PTEN fold change reported by the TSO500 pipeline
- `TCC GEPADO`: tumor cell content

Rows with a missing value in any of these columns are dropped before prediction.

## Usage

### Running Inference

The easiest way is the notebook `notebooks/inference.ipynb`.

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

If you leave out `output_dir`, `run_inference` returns the IDs, predictions and the transformed feature matrix instead of writing a file.

## Output

Predictions are written as JSON with two lists, `ids` (the `AN number` values) and `preds`:
- `output/zyg_pred/predictions.json`: three class probabilities per sample
- `output/biocall_pred/predictions.json`: one probability of wild type per sample

## Training

Training goes through `main.py`, which reads a JSON experiment config and runs an `XGBExperiment` (`src/training/train.py`). It is set up to be launched by [Syne Tune](https://github.com/syne-tune/syne-tune) during hyperparameter search. The search spaces are in `config/search_space.py`, and the hyperparameters of the selected models are in `config/config.py`.

The training CSV needs the same columns as above, plus the target (`Bioinf called as` with `WT`/`del`, or `Bioinf Zygosity` with 0/1/2) and a `Test set` column (1 for held-out samples, 0 for training).

## Citation

If you use this code, please cite:

```bibtex
@article{kerle2026pten,
  title   = {Machine learning uncovers previously undetectable genetic deletions in cancer panel sequencing},
  author  = {Kerle, Irina A. and Husak, Oleksandr and Gross, Thomas and Hartig, Andreas and Prause, Rebecca and Richter, Daniela and Br{\"u}ckmann, Sascha and Werner, Maximilian and K{\"o}gler, Anja and Hanf, Dorothea and M{\"o}hrmann, Lino and William, Doreen and Wendt, Karsten and Pfarr, Nicole and Mogler, Carolin and Aust, Daniela E. and Schr{\"o}ck, Evelin and Heining, Christoph and Glimm, Hanno},
  journal = {npj Precision Oncology},
  volume  = {10},
  number  = {1},
  pages   = {397},
  year    = {2026},
  doi     = {10.1038/s41698-026-01756-9}
}
```

The same information is in [`CITATION.cff`](CITATION.cff), so GitHub also shows a "Cite this repository" button.

## License

MIT, see [LICENSE](LICENSE).
