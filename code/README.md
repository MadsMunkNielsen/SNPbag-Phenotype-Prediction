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
- [Running Tests](#running-tests)
- [Checkpoint Layout](#checkpoint-layout)

---

## Overview

SNPbag is a transformer encoder pretrained with **masked genotype modeling** (analogous to BERT's masked language modeling) on large unlabelled genomic datasets, then fine-tuned for binary disease phenotype prediction.

The key architectural choices are:

| Design decision | Detail |
|---|---|
| Input representation | Genotype token embeddings fused with learned per-SNP identity embeddings |
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
| `--warmup-epochs` | `2` | Linear LR warmup duration |
| `--d-model` | `512` | Transformer hidden dimension |
| `--n-layers` | `16` | Number of encoder layers |
| `--n-heads` | `16` | Attention heads |
| `--window-size` | `256` | Local attention window size (`None` = full attention) |
| `--epochs` | `20` | Training epochs |
| `--batch-size` | `16` | Batch size |

---

### `finetune_phenotype.py`

CLI script for fine-tuning the pretrained encoder on a labelled phenotype. Also houses shared data-pipeline utilities used by `eval_test.py` and the test suite.

**What it does:** reads the binary case/control phenotype from column 6 of the PLINK `.fam` file (as written by `simulate_pheno_cpbayes.R`), attaches a classification head to the pretrained encoder, and fine-tunes with binary cross-entropy. Optionally freezes the encoder (transfer learning) or unfreezes it end-to-end. Saves the best checkpoint, a per-epoch history CSV, and a predictions CSV for downstream ROC analysis.

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
| `--plink-prefix` | `Data/NewSyn` | Path without extension |
| `--phenotype-kind` | `"phenotype"` | Label used in checkpoint directory and output filenames |
| `--unfreeze-encoder` | off | Fine-tune encoder weights as well |
| `--encoder-lr` | `1e-5` | Learning rate for encoder |
| `--head-lr` | `1e-4` | Learning rate for classification head |
| `--max-snps` | all | Truncate to first N SNPs |
| `--subject-sizes` | — | Sweep over these training-set sizes (space-separated) |
| `--analysis-dir` | `checkpoints/analysis` | Output directory for sweep summary CSVs |
| `--use-wandb` | off | Log metrics to Weights & Biases |

**Utility functions (also importable):**

| Symbol | Role |
|---|---|
| `PhenotypeDataset` | PyTorch `Dataset` wrapping genotype tensors and phenotype labels |
| `impute_missing_dosages()` | Imputes missing genotypes using per-SNP column means |
| `split_individuals()` | Deterministic train / validation / test split |
| `build_phenotype_loaders()` | Returns `DataLoader` triplet ready for fine-tuning |
| `classification_metrics()` | Computes accuracy and AUC using `sklearn.metrics.roc_auc_score` |
| `regression_metrics()` | Computes MSE and R² |

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

## Running Tests

```bash
# From the code/ directory — pytest.ini configures testpaths automatically
.venv/bin/pytest
.venv/bin/pytest -v    # verbose
```

Tests use small synthetic tensors and do not require PLINK files or a GPU. They cover the model forward pass, dataset tokenisation, phenotype utilities, and visualization helpers.

---

## Checkpoint Layout

```
SNPbag/checkpoints/
├── pretrain/
│   ├── {prefix}_n{N}_snps{S}_seed{seed}_best.pt
│   └── {prefix}_n{N}_snps{S}_seed{seed}_history.csv
├── finetune_cpbayes/
│   └── {phenotype-kind}-{frozen|tuned}-seed{seed}/
│       ├── best.pt
│       ├── history.csv
│       └── predictions.csv
└── analysis/
    ├── subject_size_summary.csv
    └── roc_curves.csv
```

Checkpoint names are generated automatically from `--save-prefix`, individual count, SNP count, and seed. Each `*_best.pt` contains the model state dict saved at the lowest validation loss. The companion `*_history.csv` records epoch-level metrics, used by `PlotPreTrain.R`.

---

*SNPbag · Master's Thesis · 2026*
