library(bigstatsr)
library(bigsnpr)

# ------------------------------------------------------------
# 0. Settings
# ------------------------------------------------------------
DATA_DIR     <- "SNPbag/TestData"
plink_name   <- "NewSyn_100k"
plink_prefix <- file.path(DATA_DIR, plink_name)

MAX_SNPS    <- 14000   # number of SNPs to use
N_CAUSAL    <- 500     # number of causal SNPs
SIGMA_BETA  <- 0.5     # std of log odds ratios (effect size scale)
PREVALENCE  <- 0.10    # target disease prevalence (CPBayes uses 10%)
SEED        <- 42

set.seed(SEED)

# ------------------------------------------------------------
# 1. Load bigSNP
# ------------------------------------------------------------
rds_file <- paste0(plink_prefix, ".rds")
if (!file.exists(rds_file)) {
  cat("Converting PLINK to bigSNP...\n")
  snp_readBed(paste0(plink_prefix, ".bed"), backingfile = plink_prefix)
}
obj <- snp_attach(rds_file)
G   <- obj$genotypes
fam <- obj$fam
map <- obj$map
names(map) <- c("chr", "rsid", "gen_pos", "pos", "a1", "a0")

n_ind <- nrow(G)
n_snp <- min(ncol(G), MAX_SNPS)

cat("Individuals:", n_ind, "\n")
cat("SNPs used:", n_snp, "\n")

# ------------------------------------------------------------
# 2. Pick causal SNPs and draw log odds ratios
# ------------------------------------------------------------
causal_idx <- sort(sample(seq_len(n_snp), size = N_CAUSAL))

# Log odds ratios drawn from N(0, sigma^2_beta).
# These are effects on the LOGIT scale, consistent with the
# logistic disease model in Majumdar et al. (2018).
beta_causal <- rnorm(N_CAUSAL, mean = 0, sd = SIGMA_BETA)

cat("Causal SNPs:", N_CAUSAL, "\n")
cat("Log OR std:", SIGMA_BETA, "\n")

# ------------------------------------------------------------
# 3. Compute the linear predictor eta_i = X_causal %*% beta
# ------------------------------------------------------------
# big_prodVec returns G_{causal} %*% beta_causal of length n_ind
eta_genetic <- big_prodVec(G, beta_causal, ind.col = causal_idx)

# ------------------------------------------------------------
# 4. Calibrate the intercept alpha so the population disease
#    prevalence equals K = PREVALENCE.
#    Solve for alpha such that:  mean( 1/(1+exp(-(alpha + eta))) ) = K
# ------------------------------------------------------------
target_prevalence <- function(alpha, eta, K) {
  mean(1 / (1 + exp(-(alpha + eta)))) - K
}

# Bisection over alpha in a wide range
alpha_hat <- uniroot(
  target_prevalence,
  interval = c(-30, 30),
  eta      = eta_genetic,
  K        = PREVALENCE,
  tol      = 1e-6
)$root

cat("Calibrated intercept alpha:", round(alpha_hat, 4), "\n")

# ------------------------------------------------------------
# 5. Compute disease probability and sample phenotype
# ------------------------------------------------------------
eta_full <- alpha_hat + eta_genetic
p_disease <- 1 / (1 + exp(-eta_full))

cat("Mean P(D=1):", round(mean(p_disease), 4), "\n")
cat("Quantiles of P(D=1):\n")
print(round(quantile(p_disease, c(0.01, 0.25, 0.5, 0.75, 0.99)), 4))

# Bernoulli draws
pheno_bin <- rbinom(n_ind, size = 1, prob = p_disease)

cat("\nCases:",    sum(pheno_bin == 1),
    "  Controls:", sum(pheno_bin == 0), "\n")
cat("Realised prevalence:", round(mean(pheno_bin), 4), "\n")

# ------------------------------------------------------------
# 6. Write phenotype files
#    (a) 0/1 coded:   used by LDpred2 R script and SNPbag
#    (b) 1/2 coded:   PLINK convention (1=control, 2=case)
# ------------------------------------------------------------
pheno_df_01 <- data.frame(
  FID   = fam$family.ID,
  IID   = fam$sample.ID,
  PHENO = pheno_bin
)
pheno_out_01 <- file.path(DATA_DIR, "simulated_cpbayes_01.pheno")
write.table(pheno_df_01, pheno_out_01,
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")
cat("Phenotype (0/1) written to:", pheno_out_01, "\n")

pheno_df_12 <- data.frame(
  FID   = fam$family.ID,
  IID   = fam$sample.ID,
  PHENO = pheno_bin + 1L   # 0->1 control, 1->2 case
)
pheno_out_12 <- file.path(DATA_DIR, "simulated_cpbayes_12.pheno")
write.table(pheno_df_12, pheno_out_12,
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")
cat("Phenotype (1/2 PLINK) written to:", pheno_out_12, "\n")

# ------------------------------------------------------------
# 7. Write the .fam file with the phenotype in column 6
#    (so finetune_phenotype.py can read it directly)
# ------------------------------------------------------------
fam_out <- data.frame(
  FID    = fam$family.ID,
  IID    = fam$sample.ID,
  PID    = fam$paternal.ID,
  MID    = fam$maternal.ID,
  Sex    = fam$sex,
  Pheno  = pheno_bin + 1L   # PLINK convention: 1=control, 2=case
)
fam_with_pheno <- file.path(DATA_DIR, paste0(plink_name, "_cpbayes.fam"))
write.table(fam_out, fam_with_pheno,
            row.names = FALSE, col.names = FALSE, quote = FALSE, sep = " ")
cat("FAM with phenotype written to:", fam_with_pheno, "\n")

# ------------------------------------------------------------
# 8. Write ground-truth causal variants
# ------------------------------------------------------------
causal_df <- data.frame(
  rsid       = map$rsid[causal_idx],
  chr        = map$chr[causal_idx],
  pos        = map$pos[causal_idx],
  a1         = map$a1[causal_idx],
  a0         = map$a0[causal_idx],
  beta_logor = beta_causal,
  col_index  = causal_idx
)
causal_out <- file.path(DATA_DIR, "simulated_cpbayes_causal.tsv")
write.table(causal_df, causal_out,
            row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")
cat("Causal variants written to:", causal_out, "\n")

# ------------------------------------------------------------
# 9. Write train / val / test split files
# ------------------------------------------------------------
set.seed(SEED + 1)
perm <- sample(n_ind)
n_train <- round(0.70 * n_ind)
n_val   <- round(0.10 * n_ind)
ids_train <- perm[1:n_train]
ids_val   <- perm[(n_train + 1):(n_train + n_val)]
ids_test  <- perm[(n_train + n_val + 1):n_ind]

write.table(
  data.frame(FID = fam$family.ID[ids_train], IID = fam$sample.ID[ids_train]),
  file.path(DATA_DIR, "train.keep"),
  row.names = FALSE, col.names = FALSE, quote = FALSE, sep = "\t"
)
write.table(
  data.frame(FID = fam$family.ID[ids_val], IID = fam$sample.ID[ids_val]),
  file.path(DATA_DIR, "val.keep"),
  row.names = FALSE, col.names = FALSE, quote = FALSE, sep = "\t"
)
write.table(
  data.frame(FID = fam$family.ID[ids_test], IID = fam$sample.ID[ids_test]),
  file.path(DATA_DIR, "test.keep"),
  row.names = FALSE, col.names = FALSE, quote = FALSE, sep = "\t"
)
cat("Train/val/test keep files written.\n")
cat("  N train:", length(ids_train), "\n")
cat("  N val:  ", length(ids_val), "\n")
cat("  N test: ", length(ids_test), "\n")

cat("\n=== Done simulating phenotype ===\n")
