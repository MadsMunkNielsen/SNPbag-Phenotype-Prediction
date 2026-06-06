library(bigstatsr)
library(bigsnpr)
library(bigreadr)
library(dplyr)
library(pROC)

# ------------------------------------------------------------
# 0. Settings
# ------------------------------------------------------------
DATA_DIR     <- "SNPbag/TestData"
plink_name   <- "NewSyn_100k"
plink_prefix <- file.path(DATA_DIR, plink_name)

bed_file <- paste0(plink_prefix, ".bed")
rds_file <- paste0(plink_prefix, ".rds")
bk_file  <- paste0(plink_prefix, ".bk")

# GWAS output produced by run_pipeline.sh
gwas_candidates <- c(
  file.path(DATA_DIR, "gwas_cpbayes_train.PHENO.glm.linear"),
  file.path(DATA_DIR, "gwas_cpbayes_train.PHENO.glm.logistic.hybrid"),
  file.path(DATA_DIR, "gwas_cpbayes_train.PHENO.glm.logistic")
)
gwas_file <- NULL
for (f in gwas_candidates) {
  if (file.exists(f)) { gwas_file <- f; break }
}
if (is.null(gwas_file)) {
  stop("No GWAS output found. Expected one of:\n",
       paste(gwas_candidates, collapse = "\n"))
}
is_logistic <- grepl("logistic", gwas_file)

train_keep_file <- file.path(DATA_DIR, "train.keep")
val_keep_file   <- file.path(DATA_DIR, "val.keep")
test_keep_file  <- file.path(DATA_DIR, "test.keep")
pheno_file      <- file.path(DATA_DIR, "simulated_cpbayes_01.pheno")

MAX_SNPS <- 14000

pred_out_file <- file.path(DATA_DIR, "ldpred2_test_predictions.csv")
roc_out_file  <- file.path(DATA_DIR, "ldpred2_roc.csv")
summary_file  <- file.path(DATA_DIR, "ldpred2_summary.csv")

NCORES <- 1
options(bigstatsr.ncores = NCORES)

cat("GWAS file:", gwas_file, "\n")
cat("GWAS type:", ifelse(is_logistic, "LOGISTIC", "LINEAR"), "\n")

# ------------------------------------------------------------
# 1. Load bigSNP
# ------------------------------------------------------------
if (!file.exists(rds_file) || !file.exists(bk_file)) {
  cat("Converting PLINK files to bigSNP format...\n")
  snp_readBed(bed_file, backingfile = plink_prefix)
}
obj <- snp_attach(rds_file)
G   <- obj$genotypes
fam <- obj$fam
map <- obj$map[-3]
names(map) <- c("chr", "rsid", "pos", "a1", "a0")

if (nrow(map) > MAX_SNPS) map <- map[1:MAX_SNPS, ]
cat("SNPs in use:", nrow(map), "\n")

# ------------------------------------------------------------
# 2. Read GWAS summary statistics
# ------------------------------------------------------------
sumstats <- fread2(gwas_file)
cat("GWAS columns:", paste(names(sumstats), collapse = ", "), "\n")

if (is_logistic) {
  ss <- sumstats %>%
    filter(TEST == "ADD") %>%
    filter(!is.na(OR), !is.na(`LOG(OR)_SE`), !is.na(OBS_CT)) %>%
    transmute(
      chr     = as.integer(`#CHROM`),
      pos     = as.integer(POS),
      rsid    = ID,
      a0      = OMITTED,
      a1      = A1,
      beta    = log(OR),
      beta_se = `LOG(OR)_SE`,
      n_eff   = OBS_CT
    )
} else {
  ss <- sumstats %>%
    filter(TEST == "ADD") %>%
    filter(!is.na(BETA), !is.na(SE), !is.na(OBS_CT)) %>%
    transmute(
      chr     = as.integer(`#CHROM`),
      pos     = as.integer(POS),
      rsid    = ID,
      a0      = OMITTED,
      a1      = A1,
      beta    = BETA,
      beta_se = SE,
      n_eff   = OBS_CT
    )
}

# For binary traits: effective sample size = 4 / (1/n_case + 1/n_ctrl)
pheno_df   <- read.table(pheno_file, header = TRUE, stringsAsFactors = FALSE)
train_keep <- read.table(train_keep_file, header = FALSE, stringsAsFactors = FALSE)
names(train_keep) <- c("FID", "IID")
train_pheno <- merge(train_keep, pheno_df, by = c("FID", "IID"))
n_case <- sum(train_pheno$PHENO == 1)
n_ctrl <- sum(train_pheno$PHENO == 0)
n_eff  <- 4 / (1/n_case + 1/n_ctrl)
ss$n_eff <- n_eff
cat("Training set: n_case =", n_case, " n_ctrl =", n_ctrl,
    " n_eff =", round(n_eff), "\n")
cat("Variants in sumstats after filtering:", nrow(ss), "\n")

# ------------------------------------------------------------
# 3. Match variants
# ------------------------------------------------------------
df_beta <- snp_match(ss, map)
cat("Matched variants:", nrow(df_beta), "\n")
if (nrow(df_beta) == 0) stop("No variants matched.")

# ------------------------------------------------------------
# 4. Read keep files and map to row indices
# ------------------------------------------------------------
read_keep <- function(file) {
  x <- read.table(file, header = FALSE, stringsAsFactors = FALSE)
  names(x) <- c("FID", "IID")[1:ncol(x)]
  x[, 1:2]
}

fam_ids <- data.frame(
  FID = fam$family.ID, IID = fam$sample.ID,
  row_index = seq_len(nrow(fam)), stringsAsFactors = FALSE
)
ind_train <- inner_join(read_keep(train_keep_file), fam_ids, by = c("FID", "IID"))$row_index
ind_val   <- inner_join(read_keep(val_keep_file),   fam_ids, by = c("FID", "IID"))$row_index
ind_test  <- inner_join(read_keep(test_keep_file),  fam_ids, by = c("FID", "IID"))$row_index
cat("N train:", length(ind_train),
    " N val:", length(ind_val),
    " N test:", length(ind_test), "\n")

# ------------------------------------------------------------
# 5. Load phenotype aligned to fam row order
# ------------------------------------------------------------
pheno_merged <- merge(fam_ids, pheno_df, by = c("FID", "IID"), all.x = TRUE)
pheno_merged <- pheno_merged[order(pheno_merged$row_index), ]
y      <- pheno_merged$PHENO
y_val  <- y[ind_val]
y_test <- y[ind_test]
cat("Phenotype - cases:", sum(y == 1), "  controls:", sum(y == 0), "\n")

# ------------------------------------------------------------
# 6. LD correlation matrix
# ------------------------------------------------------------
cat("Computing LD correlation matrix...\n")
# size parameter is in kb when positions are converted; using 500kb window.
pos_kb <- map$pos[df_beta$`_NUM_ID_`] / 1000
corr <- snp_cor(
  G,
  ind.row   = ind_train,
  ind.col   = df_beta$`_NUM_ID_`,
  infos.pos = pos_kb,
  size      = 500
)
corr <- as_SFBM(corr)

# ------------------------------------------------------------
# 7. LDpred2-auto with stability settings
# ------------------------------------------------------------
n_matched   <- nrow(df_beta)
n_simulated <- MAX_SNPS
H2_INIT     <- 0.5 * (n_matched / n_simulated)
cat("h2_init (scaled to matched variants):", round(H2_INIT, 4), "\n")

cat("Running LDpred2-auto...\n")
set.seed(1)
multi_auto <- snp_ldpred2_auto(
  corr            = corr,
  df_beta         = df_beta,
  h2_init         = H2_INIT,
  vec_p_init      = seq_log(1e-4, 0.2, length.out = 30),
  burn_in         = 500,
  num_iter        = 500,
  sparse          = FALSE,
  allow_jump_sign = FALSE,
  shrink_corr     = 0.95,
  use_MLE         = FALSE
)

beta_auto <- sapply(multi_auto, function(auto) auto$beta_est)
chain_ok  <- apply(beta_auto, 2, function(b) !all(is.na(b)))
cat("Converged chains:", sum(chain_ok), "/", length(chain_ok), "\n")
if (sum(chain_ok) == 0) stop("All LDpred2-auto chains failed.")
beta_auto <- beta_auto[, chain_ok, drop = FALSE]

# Per-chain h2 and p estimates (diagnostic)
h2_est_chains <- sapply(multi_auto[chain_ok], function(a) tail(a$path_h2_est, 1))
p_est_chains  <- sapply(multi_auto[chain_ok], function(a) tail(a$path_p_est, 1))
cat("h2 estimates (mean/min/max):",
    round(mean(h2_est_chains), 4), "/",
    round(min(h2_est_chains), 4),  "/",
    round(max(h2_est_chains), 4), "\n")
cat("p  estimates (mean/min/max):",
    round(mean(p_est_chains), 6), "/",
    round(min(p_est_chains), 6),  "/",
    round(max(p_est_chains), 6), "\n")

# ------------------------------------------------------------
# 8. Select best chain by validation AUC
# ------------------------------------------------------------
prs_val <- big_prodMat(
  G, beta_auto,
  ind.row = ind_val,
  ind.col = df_beta$`_NUM_ID_`
)
val_auc <- apply(prs_val, 2, function(prs) {
  if (all(is.na(prs)) || sd(prs, na.rm = TRUE) == 0) return(NA_real_)
  as.numeric(auc(y_val, prs, quiet = TRUE))
})
if (all(is.na(val_auc))) stop("All validation AUC values are NA.")
best <- which.max(val_auc)
cat("Best validation AUC:", round(val_auc[best], 4), "\n")
beta_best <- beta_auto[, best]

# ------------------------------------------------------------
# 9. Test set evaluation
# ------------------------------------------------------------
cat("\n=== TEST SET RESULTS ===\n")
prs_test <- big_prodVec(
  G, beta_best,
  ind.row = ind_test,
  ind.col = df_beta$`_NUM_ID_`
)

# AUC and confidence interval
test_auc <- as.numeric(auc(y_test, prs_test, quiet = TRUE))
roc_obj  <- roc(y_test, prs_test, quiet = TRUE)
ci_obj   <- ci.auc(roc_obj)
cat("Test AUC:", round(test_auc, 4),
    " (95% CI:", round(ci_obj[1], 4), "-", round(ci_obj[3], 4), ")\n")

# Accuracy with optimal threshold selected on validation set
prs_val_best <- prs_val[, best]
thresholds   <- quantile(prs_val_best, probs = seq(0.1, 0.9, by = 0.05))
val_accs     <- sapply(thresholds, function(t) mean((prs_val_best >= t) == y_val))
opt_thresh   <- thresholds[which.max(val_accs)]
pred_class   <- as.integer(prs_test >= opt_thresh)
accuracy     <- mean(pred_class == y_test)
cat("Test accuracy:", round(accuracy, 4),
    "  (threshold tuned on validation)\n")

cat("\nConfusion matrix:\n")
print(table(Predicted = pred_class, Actual = y_test))

# ------------------------------------------------------------
# 10. Save predictions, ROC data, and summary
# ------------------------------------------------------------
# Per-individual predictions
out <- data.frame(
  FID        = fam$family.ID[ind_test],
  IID        = fam$sample.ID[ind_test],
  phenotype  = y_test,
  prs        = prs_test,
  prediction = pred_class
)
write.csv(out, pred_out_file, row.names = FALSE)
cat("\nPredictions saved to:", pred_out_file, "\n")

# ROC curve points
roc_df <- data.frame(
  FPR = rev(1 - roc_obj$specificities),
  TPR = rev(roc_obj$sensitivities)
)
write.csv(roc_df, roc_out_file, row.names = FALSE)
cat("ROC curve saved to:", roc_out_file, "\n")

# Summary
summary_df <- data.frame(
  method   = "LDpred2-auto",
  accuracy = accuracy,
  auc      = test_auc,
  auc_lo   = ci_obj[1],
  auc_hi   = ci_obj[3],
  n_test   = length(y_test)
)
write.csv(summary_df, summary_file, row.names = FALSE)
cat("Summary saved to:", summary_file, "\n")
cat("Done.\n")
