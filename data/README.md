# Drug Toxicity Prediction — Data Directory

Place the Tox21 dataset CSV file here as `tox21.csv`.

## Download Instructions

1. Go to: https://www.kaggle.com/datasets/epicskills/tox21-dataset
2. Download the dataset
3. Place `tox21.csv` (or `train.csv`) in this `data/` folder
4. Run: `python train.py`

## File Format Expected

The CSV must have these columns:
- `smiles` (or `SMILES`) — SMILES string for each compound
- `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`
- `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`

## Without Dataset

If `tox21.csv` is not found, the training script automatically generates
a **synthetic dataset** for demonstration purposes. The app will still
work fully — predictions just won't reflect real Tox21 data.
