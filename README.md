# DeepShieldIDS
### An AI-Powered Intrusion Detection System Leveraging HybridIDSNet (CNN + LSTM + Attention)

This project is a full working implementation of the **DeepShieldIDS** framework: a hybrid
deep-learning intrusion detection system that combines **CNN** (spatial feature extraction),
**LSTM** (temporal sequence modelling), and an **attention mechanism** (feature weighting)
into a single model — **HybridIDSNet** — evaluated on **CIC-IDS2017** and **UNSW-NB15**
in both independent and cross-dataset settings, and benchmarked against Random Forest,
SVM, XGBoost, CNN-only, and LSTM-only baselines.

---

## 1. Project Structure

```
DeepShieldIDS/
├── main.py                     # single entry point / CLI runner
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                    # put downloaded CSV datasets here
│   └── processed/              # scaler / label-encoder / feature-list .pkl files (auto-generated)
├── models/                     # trained model checkpoints (auto-generated)
├── outputs/                    # metrics, confusion matrices, training curves (auto-generated)
└── src/
    ├── preprocess.py           # cleaning, feature harmonisation, SMOTE, sequence reshaping
    ├── model.py                # HybridIDSNet (CNN + LSTM + Attention) in PyTorch
    ├── baselines.py             # RandomForest / SVM / XGBoost / CNN-only / LSTM-only
    ├── train.py                 # training loop with early stopping + checkpointing
    ├── evaluate.py               # accuracy / precision / recall / F1 / ROC-AUC / confusion matrix
    ├── cross_dataset_eval.py    # train on dataset A → test on B, and vice versa
    ├── compare_baselines.py     # reproduces the paper's comparison table
    └── utils.py                 # seeding, metrics, plotting helpers
```

---

## 2. Requirements

- **Python 3.9–3.11** (recommended)
- pip packages (see `requirements.txt`):
  ```
  numpy>=1.23.0
  pandas>=1.5.0
  scikit-learn>=1.2.0
  torch>=2.0.0
  imbalanced-learn>=0.10.0
  matplotlib>=3.6.0
  seaborn>=0.12.0
  tqdm>=4.64.0
  joblib>=1.2.0
  ```
- Additionally, for the baseline-comparison script only: `xgboost>=1.7.0`
- No GPU is required — the code auto-detects CUDA and falls back to CPU. A GPU (even a
  modest one) will speed up training on the full-size real datasets considerably.

### Install

```bash
# (recommended) create a virtual environment first
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install xgboost              # only needed for compare_baselines.py
```

---

## 3. Getting the Datasets

The repository does **not** bundle the real benchmark datasets (they are large and
distributed by their original authors). Download the CSV versions and place them in
`data/raw/`:

- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Both are distributed as multiple CSV files; you can concatenate the day-wise / attack-wise
CSVs into a single file per dataset, e.g.:

```bash
cd data/raw
python -c "
import pandas as pd, glob
files = glob.glob('CIC-IDS2017_*.csv')
pd.concat([pd.read_csv(f, low_memory=False) for f in files]).to_csv('CIC-IDS2017.csv', index=False)
"
```

Each dataset needs a `label` (or `Label` / `attack_cat`) column — `preprocess.py`
auto-detects common column-name variants used across public dumps of these datasets.

**You don't have to download anything to try the project** — every script also works
out-of-the-box on an auto-generated synthetic dataset (`--data` omitted), so you can verify
the whole pipeline runs before pointing it at the real benchmark files.

---

## 4. How to Run

All commands can be run either directly via `src/*.py`, or through the single
`main.py` runner shown below.

### a) Quick demo (synthetic data, no download needed)
```bash
python main.py --stage demo --epochs 20
```
This trains HybridIDSNet on a synthetic IDS-like dataset and evaluates it, so you can
confirm the environment is set up correctly.

### b) Train on a real dataset
```bash
python main.py --stage train --data data/raw/CIC-IDS2017.csv --epochs 40 --tag cic2017
```

### c) Evaluate a trained checkpoint
```bash
python main.py --stage evaluate --data data/raw/CIC-IDS2017.csv \
    --checkpoint models/hybridids_best.pt --tag cic2017
```

### d) Cross-dataset generalisation (train on A, test on B, and vice versa)
```bash
python main.py --stage cross_dataset \
    --dataset_a data/raw/CIC-IDS2017.csv \
    --dataset_b data/raw/UNSW-NB15.csv \
    --epochs 30
```

### e) Compare HybridIDSNet against classical ML + standalone DL baselines
```bash
python main.py --stage compare --data data/raw/CIC-IDS2017.csv --epochs 20 --tag cic2017
```

Outputs (metrics JSON, confusion matrix PNG, training-curve PNG) are written to `outputs/`,
and model checkpoints to `models/`.

---

## 5. What Each Script Does

| Script | Purpose |
|---|---|
| `preprocess.py` | Loads raw CSVs, removes NaN/Inf/duplicates, label-encodes classes, standard-scales features, applies SMOTE oversampling to the training split, reshapes tabular rows into `(seq_len, 1)` pseudo-sequences for the CNN-LSTM input, and (for cross-dataset mode) restricts both datasets to their common numeric feature columns. |
| `model.py` | Defines `HybridIDSNet`: two Conv1D+BatchNorm+MaxPool blocks → BiLSTM → additive (Bahdanau-style) attention pooling → dense softmax classifier. |
| `baselines.py` | Random Forest / SVM / XGBoost wrappers, plus CNN-only and LSTM-only PyTorch ablation models for comparison. |
| `train.py` | Trains a model with Adam + `ReduceLROnPlateau`, early stopping on validation loss, and saves the best checkpoint + loss/accuracy curves. |
| `evaluate.py` | Computes accuracy, precision, recall, F1, and ROC-AUC (multi-class one-vs-rest), plus a confusion-matrix plot and a full `sklearn` classification report. |
| `cross_dataset_eval.py` | Implements the paper's two-way cross-dataset validation protocol. |
| `compare_baselines.py` | Trains every model on an identical split and prints/saves a side-by-side comparison table. |

---

## 6. Notes, Caveats & Honest Limitations

This code faithfully implements the **architecture and evaluation protocol** described in
the accompanying paper. A few things worth knowing before using it for a thesis/viva
defense or production system:

- **Reported numbers depend on the data you use.** The synthetic demo dataset is only for
  smoke-testing the pipeline — it is *not* a substitute for CIC-IDS2017 / UNSW-NB15, and
  its accuracy numbers should not be quoted as reproducing the paper's results. You need
  the real datasets to reproduce paper-level accuracy figures.
- **No adversarial robustness testing is implemented** (evasion attacks, poisoning, or
  adaptive adversaries) — consistent with the paper's own stated limitations. Only
  supervised benchmark accuracy and cross-dataset generalisation are measured.
- **No real-time deployment benchmarking** (latency/throughput under production load) is
  included; this is a research/training pipeline, not a deployed network appliance.
- Feature-intersection cross-dataset alignment is a simple *column-name overlap* strategy —
  for CIC-IDS2017 vs UNSW-NB15 in practice you may need to manually rename a handful of
  semantically-equivalent columns (e.g. duration/byte-count fields) so they overlap
  correctly, since the two datasets don't use identical column naming conventions.

---

## 7. Citation

If you use this implementation in your report, cite the original paper this code
implements:

> Mubeen, S., & Gudla, B. *DeepShieldIDS: An AI-Powered Intrusion Detection System
> Leveraging HybridIDSNet for Robust Network Security.*
