# SNPbag — Transformer-Based Polygenic Risk Scoring

> Master's Thesis · Mads Munk · 2026
>
> A BERT-style masked genotype model for polygenic risk score (PRS) prediction,
> benchmarked against LDpred2 on simulated population-genetic datasets.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Python Package — `SNPbag/`](#python-package--snpbag)
- [R Analysis Scripts](#r-analysis-scripts)
- [Generated Plots — `Plots/`](#generated-plots--plots)
- [Server Setup Guide](#server-setup-guide)
  - [1. Environment](#1-environment)
  - [2. Python Dependencies](#2-python-dependencies)
  - [3. R Dependencies](#3-r-dependencies)
  - [4. External Tools](#4-external-tools)
  - [5. Data Preparation](#5-data-preparation)
  - [6. Pretraining](#6-pretraining)
  - [7. Fine-Tuning](#7-fine-tuning)
  - [8. Evaluation](#8-evaluation)
  - [9. Full Pipeline](#9-full-pipeline)
- [Running Tests](#running-tests)
- [Checkpoint Layout](#checkpoint-layout)

---

## Overview

SNPbag is a transformer encoder pretrained with **masked genotype modeling** (analogous to BERT's masked language modeling) on large unlabelled genomic datasets, then fine-tuned for binary disease phenotype prediction.

The key architectural choices are:

| Design decision | Detail |
|---|---|
| Input representation | Genotype token embeddings fused with learned per-SNP identity embeddings |
| Attention mechanism | Sliding-window local attention — O(L × w) instead of O(L²) |
| Pretraining objective | Masked genotype reconstruction (mask probability 0.85) |
| Downstream task | Binary phenotype classification (ROC/AUC) |
| Baseline comparator | LDpred2-auto (Gibbs-sampler polygenic risk score) |

Phenotypes are simulated under the **CPBayes logistic model** (Majumdar et al., 2018): ~500 causal SNPs, log-odds ratios drawn from a normal prior, 10 % disease prevalence.

---

## Repository Structure

```
code/
├── SNPbag/                        # Python package (model, data, training)
│   ├── model.py
│   ├── dataset.py
│   ├── phenotype.py
│   ├── train_pretrain.py
│   ├── finetune_phenotype.py
│   ├── eval_test.py
│   ├── visualization.py
│   ├── roc_curve_data.csv         # ROC evaluation output
│   ├── Tests/
│   │   ├── conftest.py
│   │   └── test_smoke.py
│   ├── Data/                      # PLINK filesets (not tracked in git — ~2.7 GB)
│   ├── checkpoints/
│   │   ├── pretrain/              # Pretrained encoder weights + history CSVs
│   │   ├── finetune_cpbayes/      # Fine-tuned classifier weights
│   │   └── analysis/              # Per-run metric summaries
├── Plots/                         # Publication-ready figures (PNG)
├── pytest.ini                     # Test configuration (testpaths = SNPbag/Tests)
├── requirements.txt               # Python dependencies
├── run_pipeline.sh                # End-to-end evaluation pipeline
├── simulate_pheno_cpbayes.R       # Phenotype simulation
├── LDPRED2.R                      # LDpred2-auto pipeline
├── EvalLDPRED2.R                  # LDpred2 evaluation & ROC
├── Visualization.R                # Aggregated thesis figures
├── PlotGeLu.R                     # GELU activation function figure
├── PlotPCA.R                      # Population structure PCA
├── PlotPreTrain.R                 # Pretraining validation-loss curves
├── PlotRocAuc.R                   # ROC/AUC for GLM baseline
├── PlotRocAuc_SNPbag.R            # ROC/AUC for SNPbag
└── PlotComparison.R               # Side-by-side LDpred2 vs SNPbag ROC
```

---

## Python Package — `SNPbag/`

### `model.py`

Defines all PyTorch neural-network components.

| Class / function | Role |
|---|---|
| `GenoSnpEmbedding` | Combines dosage token embeddings with learned SNP-identity embeddings (replaces sinusoidal positional encoding, since genomic loci have no meaningful order beyond linkage) |
| `SlidingWindowEncoderLayer` | Transformer encoder layer using local sliding-window attention for linear-time scaling with sequence length |
| `AttentionEncoderLayer` | Standard multi-head self-attention layer with optional per-head weight capture (used for analysis) |
| `Encoder` | Full stack of encoder layers |
| `MLPDecoder` | MLP head for masked-token reconstruction |
| `SNPbagForMaskedGenotypeModeling` | Complete pretraining model: embedding → encoder → decoder |
| `PhenotypeRegressor` | Fine-tuning model: frozen/unfrozen encoder → classification head |
| `build_snpbag_pretrain` | Factory function that wires up a pretrain model from hyperparameters |
| `build_phenotype_regressor` | Factory that attaches a classification head to a pretrained encoder |
| `load_pretrained_encoder` | Loads encoder weights from a `.pt` checkpoint |

---

### `dataset.py`

Handles all PLINK genomic data loading and tokenisation.

| Symbol | Role |
|---|---|
| `load_plink()` | Reads PLINK binary filesets (`.bed` / `.bim` / `.fam`) via `pyplink` |
| `encode_genotypes()` | Maps PLINK dosages `{0, 1, 2, −1}` to token IDs `{0, 1, 2, 3=missing, 4=mask}` |
| `encode_snp_ids()` | Assigns stable 0-based integer IDs to SNPs for the identity embedding lookup |
| `MaskedGenotypeDataset` | PyTorch `Dataset` that randomly masks genotypes at pretraining time |
| `NUM_GENO_TOKENS = 5` | Vocabulary size for the input embedding (0, 1, 2, missing, mask) |
| `NUM_GENOTYPE_CLASSES = 3` | Output classes for the decoder (dosage 0 / 1 / 2) |

---

### `phenotype.py`

Phenotype generation, data splitting, and evaluation metrics.

| Symbol | Role |
|---|---|
| `generate_synthetic_phenotype()` | Generates continuous phenotypes under three architectures: **linear**, **interaction** (pairwise SNP products), **nonlinear** (squared + sinusoidal) |
| `PhenotypeDataset` | PyTorch `Dataset` wrapping genotype tensors and phenotype labels |
| `impute_missing_dosages()` | Imputes missing genotypes using per-SNP column means |
| `split_individuals()` | Deterministic train / validation / test split |
| `build_phenotype_loaders()` | Returns `DataLoader` triplet ready for fine-tuning |
| `classification_metrics()` | Computes accuracy and AUC using `sklearn.metrics.roc_auc_score` |
| `binarize_phenotype()` | Thresholds a continuous phenotype at zero for binary classification |
| `standardize_tensor()` | Z-score standardisation |

---

### `train_pretrain.py`

CLI script for pretraining SNPbag on unlabelled genotype data.

**What it does:** loads a PLINK fileset, constructs a `MaskedGenotypeDataset`, builds the transformer, and trains with cross-entropy loss on masked tokens. Saves best checkpoint and per-epoch history CSV.

```bash
# Full-scale pretraining
python SNPbag/train_pretrain.py \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --max-snps 14000 \
  --max-individuals 100000 \
  --d-model 512 \
  --n-layers 16 \
  --n-heads 16 \
  --d-ff 2048 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --warmup-epochs 5 \
  --save-dir SNPbag/checkpoints/pretrain \
  --save-prefix snpbag_n100k
```

| Key argument | Default | Description |
|---|---|---|
| `--plink-prefix` | `Data/NewSyn` | Path without extension |
| `--max-snps` | all | Truncate to first N SNPs |
| `--max-individuals` | all | Subsample to N individuals |
| `--mask-prob` | `0.85` | Fraction of tokens masked per sample |
| `--d-model` | `512` | Transformer hidden dimension |
| `--n-layers` | `16` | Number of encoder layers |
| `--n-heads` | `16` | Attention heads |
| `--window-size` | `256` | Sliding-window half-width |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `16` | Batch size |

---

### `finetune_phenotype.py`

CLI script for fine-tuning the pretrained encoder on a labelled phenotype.

**What it does:** loads a pretrained checkpoint, attaches a classification head, and fine-tunes with binary cross-entropy. Optionally freezes the encoder (transfer learning) or unfreezes it end-to-end. Generates ROC curve data and a summary CSV.

```bash
python SNPbag/finetune_phenotype.py \
  --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_n100000_snps14000_seed42_best.pt \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --phenotype-kind linear \
  --epochs 30 \
  --encoder-lr 1e-5 \
  --head-lr 1e-4 \
  --max-snps 14000 \
  --save-dir SNPbag/checkpoints/finetune_cpbayes
```

| Key argument | Default | Description |
|---|---|---|
| `--pretrained-checkpoint` | required | Path to `.pt` pretrain checkpoint |
| `--phenotype-kind` | `None` | `linear`, `interaction`, or `nonlinear` |
| `--unfreeze-encoder` | off | Fine-tune encoder weights as well |
| `--encoder-lr` | `1e-5` | Learning rate for encoder |
| `--head-lr` | `1e-4` | Learning rate for classification head |
| `--use-wandb` | off | Log metrics to Weights & Biases |

---

### `eval_test.py`

Lightweight evaluation script. Loads a fine-tuned checkpoint, runs inference on the held-out test split, and writes predicted probabilities plus ground-truth labels for downstream ROC analysis.

---

### `visualization.py`

Utility module (not a standalone script) providing:

- `collect_masked_prediction_stats()` — per-token and per-position reconstruction accuracy during pretraining
- `save_masked_prediction_analysis()` — writes analysis to CSV
- `plot_phenotype_summary()` — ROC curve + prediction-distribution figure

---

### `Tests/test_smoke.py`

Unit smoke tests covering all core modules. Uses tiny synthetic tensors — no PLINK files or GPU required.

```bash
# From the code/ directory — pytest.ini points to SNPbag/Tests automatically
pytest
pytest -v    # verbose
```

---

## R Analysis Scripts

### `simulate_pheno_cpbayes.R`

Simulates a binary case/control phenotype following the CPBayes logistic model:

1. Selects ~500 causal SNPs at random
2. Draws log-odds ratios from N(0, σ²_β)
3. Computes linear predictor η_i = α + Σ β_m · G_{im}
4. Samples case status with P(D=1 | G_i) = logistic(η_i)
5. Adjusts intercept α to enforce 10 % prevalence

**Output:** PLINK-format `.pheno` file (columns: FID, IID, PHENO with 1 = control / 2 = case).

**Dependencies:** `bigstatsr`, `bigsnpr`

---

### `LDPRED2.R`

Complete LDpred2-auto pipeline used as the statistical baseline:

1. Loads PLINK binary + phenotype
2. Runs linear-regression GWAS on the training split via PLINK2
3. Computes an LD correlation matrix
4. Runs the LDpred2-auto Gibbs sampler (multiple independent chains)
5. Selects the best chain by validation AUC
6. Computes a genome-wide polygenic risk score (PRS)
7. Evaluates on the held-out test set and writes a predictions CSV

**Dependencies:** `bigstatsr`, `bigsnpr`, `bigreadr`, `dplyr`, `pROC`

---

### `EvalLDPRED2.R`

Post-hoc evaluation of LDpred2 predictions. Reads the predictions CSV produced by `LDPRED2.R` and reports ROC curve, AUC, accuracy, sensitivity, specificity, and optimal Youden-J threshold.

---

### `Visualization.R`

Central figure-generation script for the thesis. Produces:

- Fine-tuning AUC by training-set size (1k → 100k individuals)
- Pretraining validation-loss curves
- Population-structure PCA
- Head-to-head model comparison plots

Uses a custom colour palette (Novo Nordisk corporate palette + extensions) and a minimal `ggplot2` theme.

---

### `PlotGeLu.R`

Renders the GELU activation function:

```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

Output: `Plots/GELU.png` (13 × 6 in, retina DPI).

---

### `PlotPCA.R`

Simulates three-population genomic data (450 individuals × 5 000 SNPs, F_ST = 0.05) and renders a PCA scatter coloured by population. Illustrates population stratification used to motivate the SNPbag architecture.

---

### `PlotPreTrain.R`

Reads per-epoch `*_history.csv` files from `checkpoints/pretrain/` and overlays validation-loss curves for all training-set sizes (1k, 5k, 10k, 50k, 100k individuals).

---

### `PlotRocAuc.R` / `PlotRocAuc_SNPbag.R`

Publication-quality ROC curve plots for the GLM baseline and SNPbag respectively. Both scripts compute the trapezoidal AUC, identify the optimal Youden-J threshold, and annotate the figure with AUC and accuracy.

---

### `PlotComparison.R`

Overlays the LDpred2 and SNPbag ROC curves on a single axis for direct comparison.

---

### `run_pipeline.sh`

Orchestrates the full experiment in sequence:

```
Step 1 — Simulate phenotype          (simulate_pheno_cpbayes.R)
Step 2 — Extract first 14 000 SNPs   (PLINK2)
Step 3 — Run GWAS on training set     (PLINK2)
Step 4 — Run LDpred2-auto             (LDPRED2.R)
Step 5 — Fine-tune SNPbag             (finetune_phenotype.py)
Step 6 — Compare on test set          (PlotComparison.R)
```

Run from the `code/` directory:

```bash
bash run_pipeline.sh
```

---

## Generated Plots — `Plots/`

| File | Contents |
|---|---|
| `GELU.png` | GELU activation function curve |
| `PCA_population_structure.png` | PCA of simulated three-population dataset |
| `ROC_GLM.png` | ROC curve for the GLM/LDpred2 baseline |
| `ROC_GLM_LDpred2.png` | ROC with LDpred2 PRS overlaid |
| `ROC_SNPbag.png` | ROC curve for SNPbag fine-tuned model |
| `ValLoss.png` | Pretraining validation-loss curves by dataset size |

---

## Server Setup Guide

The following steps assume a Linux server with SLURM or direct shell access and a GPU node.

---

### 1. Environment

```bash
git clone <repo-url> speciale && cd speciale/code

python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU servers, install PyTorch with the matching CUDA wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> Replace `cu121` with your server's CUDA version (`nvidia-smi` shows it).
> For CPU-only nodes use `--index-url https://download.pytorch.org/whl/cpu`.

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

### 3. R Dependencies

```r
install.packages(c("bigstatsr", "bigsnpr", "bigreadr", "dplyr", "ggplot2", "pROC"))
```

If `bigsnpr` cannot find PLINK2 automatically, set the path in the R scripts:

```r
options(bigsnpr.plink2.path = "/path/to/plink2")
```

---

### 4. External Tools

**PLINK2** is required for GWAS and data extraction steps.

```bash
# Download PLINK2 (Linux x86_64)
wget https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20250104.zip
unzip plink2_linux_x86_64_20250104.zip -d /usr/local/bin/
chmod +x /usr/local/bin/plink2
plink2 --version
```

---

### 5. Data Preparation

Place PLINK binary filesets (`.bed`, `.bim`, `.fam`) at:

```
SNPbag/Data/
├── NewSyn_100k_cpbayes.bed
├── NewSyn_100k_cpbayes.bim
└── NewSyn_100k_cpbayes.fam
```

> The data files are not tracked in this repository (≈2.7 GB).
> Contact the author for access or regenerate them using the simulation scripts.

If using a different path, update `PLINK_PREFIX` in `run_pipeline.sh` and pass `--plink-prefix` explicitly to the Python scripts.

---

### 6. Pretraining

Pretraining is the most GPU-intensive step. A full run on 100 k individuals and 14 k SNPs takes roughly 6–12 hours on a single A100.

```bash
cd code

python SNPbag/train_pretrain.py \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --max-snps 14000 \
  --max-individuals 100000 \
  --d-model 512 \
  --n-layers 16 \
  --n-heads 16 \
  --d-ff 2048 \
  --window-size 256 \
  --mask-prob 0.85 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --warmup-epochs 5 \
  --seed 42 \
  --num-workers 4 \
  --save-dir SNPbag/checkpoints/pretrain \
  --save-prefix snpbag_n100k_snps14k
```

The best checkpoint is saved to:

```
SNPbag/checkpoints/pretrain/snpbag_n100k_snps14k_best.pt
```

Training history (epoch, val_loss, accuracy) is written to a companion `*_history.csv`.

**SLURM example:**

```bash
#!/bin/bash
#SBATCH --job-name=snpbag-pretrain
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain_%j.out

source .venv/bin/activate

python SNPbag/train_pretrain.py \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --max-snps 14000 \
  --max-individuals 100000 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 4 \
  --save-dir SNPbag/checkpoints/pretrain
```

---

### 7. Fine-Tuning

Fine-tuning is much faster (typically < 1 hour per run).

```bash
python SNPbag/finetune_phenotype.py \
  --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_n100k_snps14k_best.pt \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --phenotype-kind linear \
  --max-snps 14000 \
  --epochs 30 \
  --batch-size 32 \
  --encoder-lr 1e-5 \
  --head-lr 1e-4 \
  --seed 42 \
  --num-workers 4 \
  --save-dir SNPbag/checkpoints/finetune_cpbayes
```

To run all three phenotype architectures in parallel:

```bash
for kind in linear interaction nonlinear; do
  python SNPbag/finetune_phenotype.py \
    --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_n100k_snps14k_best.pt \
    --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
    --phenotype-kind $kind \
    --max-snps 14000 \
    --epochs 30 \
    --save-dir SNPbag/checkpoints/finetune_cpbayes/$kind &
done
wait
```

---

### 8. Evaluation

After fine-tuning, evaluate on the held-out test split:

```bash
# R evaluation — generates Plots/ROC_SNPbag.png
Rscript PlotRocAuc_SNPbag.R

# Run LDpred2 pipeline and compare
Rscript LDPRED2.R
Rscript PlotComparison.R
```

---

### 9. Full Pipeline

To reproduce the complete experiment from scratch:

```bash
# From the code/ directory
source .venv/bin/activate
bash run_pipeline.sh
```

This runs all steps end-to-end: phenotype simulation → GWAS → LDpred2 → SNPbag fine-tune → comparison figures.

---

## Running Tests

```bash
# From the code/ directory — pytest.ini configures testpaths automatically
.venv/bin/pytest
.venv/bin/pytest -v    # verbose
```

Tests use small synthetic tensors and do not require PLINK files or a GPU. They cover the model forward pass, dataset tokenisation, phenotype utilities, and visualization helpers (58 tests total).

---

## Checkpoint Layout

```
SNPbag/checkpoints/
├── pretrain/
│   ├── snpbag_n1000_snps14000_seed42_best.pt
│   ├── snpbag_n1000_snps14000_seed42_history.csv
│   ├── snpbag_n5000_snps14000_seed42_best.pt
│   ├── snpbag_n5000_snps14000_seed42_history.csv
│   ├── snpbag_n10000_snps14000_seed42_best.pt
│   ├── snpbag_n10000_snps14000_seed42_history.csv
│   ├── snpbag_n50000_snps14000_seed42_best.pt
│   ├── snpbag_n50000_snps14000_seed42_history.csv
│   ├── snpbag_n100000_snps14000_seed42_best.pt
│   └── snpbag_n100000_snps14000_seed42_history.csv
├── finetune_cpbayes/
│   └── linear-frozen-seed42/
│       └── best.pt
└── analysis/
    └── subject_size_summary.csv
```

Each `*_best.pt` file contains the full model state dict saved at the epoch with the lowest validation loss. The companion `*_history.csv` records epoch-level `val_loss` and reconstruction accuracy, used by `PlotPreTrain.R`.

---

*SNPbag · Master's Thesis · 2026*
