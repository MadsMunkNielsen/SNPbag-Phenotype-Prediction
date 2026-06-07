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

Fine-tunes the pretrained encoder for phenotype prediction. Reads the binary case/control phenotype from column 6 of the PLINK `.fam` file. Supports frozen/unfrozen encoder, subject-size sweeps, and optional W&B logging. Also houses shared data-pipeline utilities (`PhenotypeDataset`, `split_individuals`, `classification_metrics`, etc.) used by `eval_test.py` and the test suite.

```bash
python SNPbag/finetune_phenotype.py \
  --pretrained-checkpoint SNPbag/checkpoints/pretrain/snpbag_best.pt \
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

*Adapting SNPbag for Phenotype Prediction: A Transformer-Based Approach to Polygenic Risk · Master's Thesis · Aalborg University · 2026*
