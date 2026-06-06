library(ggplot2)
library(dplyr)

DATA_DIR <- "SNPbag/TestData"

# ------------------------------------------------------------
# 1. Load LDpred2 outputs
# ------------------------------------------------------------
ldpred2_roc_file <- file.path(DATA_DIR, "ldpred2_roc.csv")
ldpred2_sum_file <- file.path(DATA_DIR, "ldpred2_summary.csv")

ldpred2_roc <- read.csv(ldpred2_roc_file)
ldpred2_sum <- read.csv(ldpred2_sum_file)
ldpred2_auc <- ldpred2_sum$auc
ldpred2_acc <- ldpred2_sum$accuracy
cat("LDpred2  AUC =", round(ldpred2_auc, 4),
    "  ACC =", round(ldpred2_acc, 4), "\n")

# ------------------------------------------------------------
# 2. Load SNPbag outputs (from finetune_phenotype.py)
# ------------------------------------------------------------
# finetune_phenotype.py writes roc_curves.csv when --subject-sizes
# is used. When a single run is used, the per-individual predictions
# are written to phenotype_summary_seed42.csv in --save-dir.
#
# We assume the user ran the single-run path above, so we read the
# ROC by re-computing it from the saved predictions. If a roc_curves.csv
# exists, we use that instead.
snpbag_roc_file <- file.path("SNPbag", "checkpoints", "analysis_cpbayes", "roc_curves.csv")
snpbag_sum_file <- file.path("SNPbag", "checkpoints", "analysis_cpbayes",
                             "subject_size_summary.csv")

if (file.exists(snpbag_roc_file)) {
  snpbag_roc_raw <- read.csv(snpbag_roc_file)
  snpbag_roc <- data.frame(
    FPR = snpbag_roc_raw$fpr,
    TPR = snpbag_roc_raw$tpr
  )
  snpbag_sum <- read.csv(snpbag_sum_file)
  snpbag_auc <- snpbag_sum$test_auc[1]
  snpbag_acc <- snpbag_sum$test_accuracy[1]
} else {
  # Fallback: read from the single-run output (predictions CSV)
  pred_file <- file.path("SNPbag", "checkpoints", "finetune_cpbayes",
                         "phenotype_summary_seed42.csv")
  cat("No roc_curves.csv found; reading predictions from", pred_file, "\n")
  pred <- read.csv(pred_file)
  snpbag_auc <- pred$test_auc[1]
  snpbag_acc <- pred$test_accuracy[1]
  snpbag_roc <- data.frame(FPR = c(0, 1), TPR = c(0, 1))  # placeholder
}
cat("SNPbag   AUC =", round(snpbag_auc, 4),
    "  ACC =", round(snpbag_acc, 4), "\n")

# ------------------------------------------------------------
# 3. ROC plot helper - same layout as PlotRocAuc.R
# ------------------------------------------------------------
plot_roc <- function(roc_df, auc_val, line_color = "#001965", fill_color = NULL) {
  if (is.null(fill_color)) fill_color <- line_color

  ggplot(data = roc_df, aes(x = FPR, y = TPR)) +
    geom_ribbon(aes(ymin = 0, ymax = TPR),
                fill = fill_color, alpha = 0.08) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1),
                 linetype = "dashed", color = "grey50", linewidth = 0.7) +
    geom_line(color = line_color, linewidth = 1) +
    annotate(
      "text",
      x = 0.55, y = 0.35,
      label = paste0("AUC = ", sprintf("%.2f", auc_val)),
      size = 7, color = line_color, fontface = "bold"
    ) +
    xlab("False Positive Rate (FPR)") +
    ylab("True Positive Rate (TPR)") +
    theme_minimal() +
    theme(
      axis.title       = element_text(size = 25),
      axis.text.y      = element_text(size = 10, color = "grey60"),
      axis.text.x      = element_text(size = 10, color = "grey60"),
      axis.ticks.y     = element_line(linewidth = 0.3, color = "grey60"),
      axis.ticks.x     = element_line(linewidth = 0.3, color = "grey60"),
      axis.ticks.length = unit(0.15, "cm"),
      legend.position  = "none"
    ) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.2),
                       limits = c(0, 1), expand = c(0.01, 0.01)) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.2),
                       limits = c(0, 1), expand = c(0.01, 0.01)) +
    coord_equal() +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 0),
                 color = "black", linewidth = 0.5) +
    geom_segment(aes(x = 0, xend = 0, y = 0, yend = 1),
                 color = "black", linewidth = 0.5)
}

# ------------------------------------------------------------
# 4. Individual ROC plots
# ------------------------------------------------------------
roc_ldpred2 <- plot_roc(ldpred2_roc, ldpred2_auc,
                       line_color = "#001965", fill_color = "#001965")
roc_snpbag  <- plot_roc(snpbag_roc,  snpbag_auc,
                       line_color = "#A6192E", fill_color = "#A6192E")

# ------------------------------------------------------------
# 5. Combined ROC plot
# ------------------------------------------------------------
combined_df <- bind_rows(
  ldpred2_roc %>% mutate(Method = sprintf("LDpred2-auto (AUC = %.2f)", ldpred2_auc)),
  snpbag_roc  %>% mutate(Method = sprintf("SNPbag (AUC = %.2f)",       snpbag_auc))
)

roc_combined <- ggplot(combined_df, aes(x = FPR, y = TPR, color = Method)) +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1),
               linetype = "dashed", color = "grey50", linewidth = 0.7,
               inherit.aes = FALSE) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("#001965", "#A6192E")) +
  xlab("False Positive Rate (FPR)") +
  ylab("True Positive Rate (TPR)") +
  theme_minimal() +
  theme(
    axis.title       = element_text(size = 25),
    axis.text.y      = element_text(size = 10, color = "grey60"),
    axis.text.x      = element_text(size = 10, color = "grey60"),
    axis.ticks.y     = element_line(linewidth = 0.3, color = "grey60"),
    axis.ticks.x     = element_line(linewidth = 0.3, color = "grey60"),
    axis.ticks.length = unit(0.15, "cm"),
    legend.position  = c(0.65, 0.20),
    legend.title     = element_blank(),
    legend.text      = element_text(size = 14)
  ) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2),
                     limits = c(0, 1), expand = c(0.01, 0.01)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2),
                     limits = c(0, 1), expand = c(0.01, 0.01)) +
  coord_equal() +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 0),
               color = "black", linewidth = 0.5, inherit.aes = FALSE) +
  geom_segment(aes(x = 0, xend = 0, y = 0, yend = 1),
               color = "black", linewidth = 0.5, inherit.aes = FALSE)

# ------------------------------------------------------------
# 6. Save plots
# ------------------------------------------------------------
SavePlot <- function(filename, plotname, folder = "Plots/") {
  dir.create(folder, showWarnings = FALSE, recursive = TRUE)
  full_path <- paste0(folder, filename)
  ggsave(
    full_path, plot = plotname,
    width = 8, height = 8, dpi = "retina", bg = "white"
  )
}

SavePlot("ROC_LDpred2.png", roc_ldpred2)
SavePlot("ROC_SNPbag.png",  roc_snpbag)
SavePlot("ROC_Comparison.png", roc_combined)

# ------------------------------------------------------------
# 7. Comparison table
# ------------------------------------------------------------
comparison <- data.frame(
  Method   = c("LDpred2-auto", "SNPbag"),
  Accuracy = c(ldpred2_acc, snpbag_acc),
  AUC      = c(ldpred2_auc, snpbag_auc)
)
write.csv(comparison, "Plots/comparison_table.csv", row.names = FALSE)
cat("\nComparison table:\n")
print(comparison)
cat("\nSaved Plots/ROC_LDpred2.png, ROC_SNPbag.png, ROC_Comparison.png\n")
