# SNPbag — Transformer-Based Polygenic Risk Scoring

> Master's Thesis · Mads Munk · Aalborg University · 2026
>
> Comparing a BERT-style masked genotype model (SNPbag) against LDpred2
> for binary disease phenotype prediction on simulated population-genetic data.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [LaTeX Source](#latex-source)
  - [`master.tex`](#mastertex)
  - [`aaumath.sty`](#aaumathsty)
  - [`incl/` — Text Content](#incl--text-content)
  - [`fig/` — Figures and Floats](#fig--figures-and-floats)
- [Code](#code)
  - [Overview](#overview)
  - [Python Package — `SNPbag/`](#python-package--snpbag)
  - [R Analysis Scripts](#r-analysis-scripts)
  - [Server Setup Guide](#server-setup-guide)
  - [Running Tests](#running-tests)
  - [Checkpoint Layout](#checkpoint-layout)

---

## Repository Structure

```
Speciale/
├── master.tex                  # Root LaTeX document
├── aaumath.sty                 # AAU mathematics style package
├── incl/                       # All written content (chapters, bib, preamble)
│   ├── main/                   # Chapter files
│   ├── app/                    # Appendices
│   ├── misc/                   # Front matter (title page, abstract, preface)
│   ├── bib/                    # Bibliography files (.bib)
│   └── pre/                    # Preamble (packages, commands, config)
├── fig/                        # Figures and floating environments
│   ├── img/                    # Image files (PNG, PDF)
│   ├── alg/                    # Algorithm pseudocode
│   ├── tab/                    # Table contents
│   └── tikz/                   # TiKZ diagrams
└── code/                       # All experiment code (Python + R)
    ├── SNPbag/                 # Python package
    ├── *.R                     # R analysis and plotting scripts
    ├── run_pipeline.sh         # End-to-end pipeline
    ├── requirements.txt        # Python dependencies
    └── pytest.ini              # Test configuration
```

---

## LaTeX Source

### `master.tex`

The root document. Imports the preamble, includes all chapter files in order, and controls the overall document structure. Compile with `pdflatex` or `latexmk`:

```bash
latexmk -pdf master.tex
```

### `aaumath.sty`

A custom LaTeX style file providing mathematical macros and notation used throughout the thesis (AAU Mathematics department conventions).

---

### `incl/` — Text Content

All written content is split into separate files and included via `\input{}` in `master.tex`.

| Folder | Contents |
|---|---|
| `incl/main/` | Chapter files: `Introdouction.tex`, `Theory.tex`, `Data.tex`, `Clinical.tex`, `SimStudy.tex`, `Discussion.tex`, `Conclusion.tex` |
| `incl/app/` | Appendix files (`AppendixA.tex`) |
| `incl/misc/` | Front matter: `frontpage.tex`, `titlepage.tex`, `abstract.tex`, `preface.tex`, `contents.tex`, `Signatures.tex` |
| `incl/bib/` | Bibliography databases: `articles.bib`, `books.bib`, `software.bib`, `references.bib` |
| `incl/pre/` | Preamble: `pkgs.tex` (package imports), `conf.tex` (document configuration), `cmds.tex` (custom commands) |

---

### `fig/` — Figures and Floats

Floating environments are kept in separate files to make the document modular and to keep individual figure changes trackable in git history.

| Folder | Contents |
|---|---|
| `fig/img/` | Raster and vector images used in figures (GELU, PCA, ROC curves, pretraining loss) |
| `fig/alg/` | Algorithm pseudocode environments |
| `fig/tab/` | Table contents |
| `fig/tikz/` | TiKZ diagram source files |

---

## Code

### Overview

SNPbag is a transformer encoder pretrained with **masked genotype modeling** (analogous to BERT's masked language modeling) on large unlabelled genomic datasets, then fine-tuned for binary disease phenotype prediction.

| Design decision | Detail |
|---|---|
| Input representation | Genotype token embeddings fused with learned per-SNP identity embeddings |
| Attention mechanism | Sliding-window local attention — O(L × w) instead of O(L²) |
| Pretraining objective | Masked genotype reconstruction (mask probability 0.85) |
| Downstream task | Binary phenotype classification (ROC/AUC) |
| Baseline comparator | LDpred2-auto (Gibbs-sampler polygenic risk score) |

Phenotypes are simulated under the **CPBayes logistic model** (Majumdar et al., 2018): ~500 causal SNPs, log-odds ratios drawn from a normal prior, 10 % disease prevalence.

---

### Python Package — `SNPbag/`

#### `model.py`

Defines all PyTorch neural-network components.

| Class / function | Role |
|---|---|
| `GenoSnpEmbedding` | Combines dosage token embeddings with learned SNP-identity embeddings (replaces sinusoidal positional encoding) |
| `SlidingWindowEncoderLayer` | Transformer encoder layer using local sliding-window attention for linear-time scaling with sequence length |
| `AttentionEncoderLayer` | Standard multi-head self-attention layer with optional per-head weight capture |
| `Encoder` | Full stack of encoder layers |
| `MLPDecoder` | MLP head for masked-token reconstruction |
| `SNPbagForMaskedGenotypeModeling` | Complete pretraining model: embedding → encoder → decoder |
| `PhenotypeRegressor` | Fine-tuning model: frozen/unfrozen encoder → classification head |
| `build_snpbag_pretrain` | Factory function that wires up a pretrain model from hyperparameters |
| `build_phenotype_regressor` | Factory that attaches a classification head to a pretrained encoder |
| `load_pretrained_encoder` | Loads encoder weights from a `.pt` checkpoint |

#### `dataset.py`

Handles all PLINK genomic data loading and tokenisation.

| Symbol | Role |
|---|---|
| `load_plink()` | Reads PLINK binary filesets (`.bed` / `.bim` / `.fam`) via `pyplink` |
| `encode_genotypes()` | Maps PLINK dosages `{0, 1, 2, −1}` to token IDs `{0, 1, 2, 3=missing, 4=mask}` |
| `encode_snp_ids()` | Assigns stable 0-based integer IDs to SNPs |
| `MaskedGenotypeDataset` | PyTorch `Dataset` that randomly masks genotypes at pretraining time |

#### `phenotype.py`

Phenotype generation, data splitting, and evaluation metrics.

| Symbol | Role |
|---|---|
| `generate_synthetic_phenotype()` | Generates phenotypes under three architectures: **linear**, **interaction**, **nonlinear** |
| `PhenotypeDataset` | PyTorch `Dataset` wrapping genotype tensors and phenotype labels |
| `split_individuals()` | Deterministic train / validation / test split |
| `build_phenotype_loaders()` | Returns `DataLoader` triplet ready for fine-tuning |
| `classification_metrics()` | Computes accuracy and AUC |
| `regression_metrics()` | Computes MSE and R² |
| `binarize_phenotype()` | Thresholds continuous phenotype at zero for binary classification |

#### `train_pretrain.py`

CLI script for pretraining SNPbag on unlabelled genotype data. Saves best checkpoint and per-epoch history CSV.

```bash
python SNPbag/train_pretrain.py \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --max-snps 14000 \
  --epochs 50 \
  --batch-size 32 \
  --save-dir SNPbag/checkpoints/pretrain
```

#### `finetune_phenotype.py`

Fine-tunes the pretrained encoder for phenotype prediction. Supports frozen/unfrozen encoder and optional W&B logging.

```bash
python SNPbag/finetune_phenotype.py \
  --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_n100000_snps14000_seed42_best.pt \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --phenotype-kind linear \
  --epochs 30 \
  --save-dir SNPbag/checkpoints/finetune_cpbayes
```

#### `eval_test.py`

Loads a fine-tuned checkpoint, runs inference on the held-out test split, and writes predicted probabilities for ROC analysis.

#### `visualization.py`

Utility module providing per-token/per-position reconstruction accuracy analysis, attention map extraction, and phenotype summary plots.

#### `Tests/test_smoke.py`

Unit smoke tests covering all core modules. Uses tiny synthetic tensors — no PLINK files or GPU required.

---

### R Analysis Scripts

| Script | Purpose |
|---|---|
| `simulate_pheno_cpbayes.R` | Simulates binary case/control phenotype (CPBayes logistic model, ~500 causal SNPs, 10 % prevalence) |
| `LDPRED2.R` | Full LDpred2-auto pipeline: GWAS → LD matrix → Gibbs sampler → PRS |
| `EvalLDPRED2.R` | Post-hoc evaluation of LDpred2 predictions (ROC, AUC, accuracy) |
| `Visualization.R` | Central figure-generation script for thesis plots |
| `PlotGeLu.R` | GELU activation function figure |
| `PlotPCA.R` | PCA of simulated three-population dataset |
| `PlotPreTrain.R` | Validation-loss curves across dataset sizes |
| `PlotRocAuc.R` | ROC/AUC figure for GLM baseline |
| `PlotRocAuc_SNPbag.R` | ROC/AUC figure for SNPbag |
| `PlotComparison.R` | Side-by-side LDpred2 vs SNPbag ROC comparison |

`run_pipeline.sh` orchestrates the full experiment: phenotype simulation → GWAS → LDpred2 → SNPbag fine-tune → comparison plots.

---

### Server Setup Guide

#### 1. Environment

```bash
git clone <repo-url> speciale && cd speciale/code

python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU servers, replace the `torch` line with a CUDA-specific wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps torch
```

> Replace `cu121` with your server's CUDA version (`nvidia-smi` shows it).

#### 3. R Dependencies

```r
install.packages(c("bigstatsr", "bigsnpr", "bigreadr", "dplyr", "ggplot2", "pROC"))
```

#### 4. External Tools

**PLINK2** is required for GWAS and data extraction steps.

```bash
wget https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_x86_64_20250104.zip
unzip plink2_linux_x86_64_20250104.zip -d /usr/local/bin/
chmod +x /usr/local/bin/plink2
```

#### 5. Data Preparation

Place PLINK binary filesets (`.bed`, `.bim`, `.fam`) at:

```
code/SNPbag/Data/
├── NewSyn_100k_cpbayes.bed
├── NewSyn_100k_cpbayes.bim
└── NewSyn_100k_cpbayes.fam
```

> The data files are not included in this repository due to their size (≈2.7 GB).
> Contact the author for access or regenerate them using the simulation scripts.

#### 6. Pretraining

```bash
cd code

python SNPbag/train_pretrain.py \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --max-snps 14000 \
  --max-individuals 100000 \
  --d-model 512 --n-layers 16 --n-heads 16 --d-ff 2048 \
  --window-size 256 --mask-prob 0.85 \
  --epochs 50 --batch-size 32 --lr 1e-4 --warmup-epochs 5 \
  --seed 42 --num-workers 4 \
  --save-dir SNPbag/checkpoints/pretrain \
  --save-prefix snpbag_n100k_snps14k
```

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
  --max-snps 14000 --max-individuals 100000 \
  --epochs 50 --batch-size 32 --num-workers 4 \
  --save-dir SNPbag/checkpoints/pretrain
```

#### 7. Fine-Tuning

```bash
python SNPbag/finetune_phenotype.py \
  --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_n100k_snps14k_best.pt \
  --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
  --phenotype-kind linear \
  --max-snps 14000 --epochs 30 --batch-size 32 \
  --encoder-lr 1e-5 --head-lr 1e-4 --seed 42 --num-workers 4 \
  --save-dir SNPbag/checkpoints/finetune_cpbayes
```

To run all three phenotype architectures in parallel:

```bash
for kind in linear interaction nonlinear; do
  python SNPbag/finetune_phenotype.py \
    --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_n100k_snps14k_best.pt \
    --plink-prefix SNPbag/Data/NewSyn_100k_cpbayes \
    --phenotype-kind $kind --max-snps 14000 --epochs 30 \
    --save-dir SNPbag/checkpoints/finetune_cpbayes/$kind &
done
wait
```

#### 8. Evaluation

```bash
# R evaluation — generates Plots/ROC_SNPbag.png
Rscript PlotRocAuc_SNPbag.R

# Run LDpred2 pipeline
Rscript LDPRED2.R

# Side-by-side comparison figure
Rscript PlotComparison.R
```

#### 9. Full Pipeline

```bash
# From the code/ directory, with .venv active
source .venv/bin/activate
bash run_pipeline.sh
```

---

### Running Tests

```bash
cd code
.venv/bin/pytest          # runs all 58 tests via pytest.ini
.venv/bin/pytest -v       # verbose output
```

Tests use small synthetic tensors and do not require PLINK files or a GPU.

---

### Checkpoint Layout

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

---

*SNPbag · Master's Thesis · Aalborg University · 2026*
