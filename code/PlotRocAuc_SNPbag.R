library(ggplot2)

# ============================================================
# ROC Curve, AUC, and Accuracy for SNPbag
# ============================================================

# --- Load ROC data from CSV ---
roc_df <- read.csv("SNPbag/checkpoints/finetune_cpbayes/roc_curve_data.csv")

# Remove infinite thresholds and sort
roc_df <- roc_df[is.finite(roc_df$Threshold), ]
roc_df <- roc_df[order(roc_df$FPR, roc_df$TPR), ]


# --- AUC via trapezoidal rule ---
auc_val <- sum(
  diff(roc_df$FPR) *
    (roc_df$TPR[-1] + roc_df$TPR[-nrow(roc_df)]) / 2
)

cat("AUC:", round(auc_val, 4), "\n")


# --- Accuracy at optimal threshold (Youden's J) ---
youden_j   <- roc_df$TPR - roc_df$FPR
best_idx   <- which.max(youden_j)
best_thresh <- roc_df$Threshold[best_idx]


# Approximate: Accuracy = TPR * prevalence + (1 - FPR) * (1 - prevalence)
# For balanced classes (prevalence = 0.5):
best_tpr <- roc_df$TPR[best_idx]
best_fpr <- roc_df$FPR[best_idx]
accuracy <- (best_tpr + (1 - best_fpr)) / 2

cat("Optimal threshold:", round(best_thresh, 4), "\n")
cat("Accuracy:        ", round(accuracy, 4), "\n")


# --- Plot ROC curve ---
ROCPlot_SNPbag <- ggplot(data = roc_df, aes(x = FPR, y = TPR)) +
  geom_ribbon(aes(ymin = 0, ymax = TPR),
              fill = "#001965", alpha = 0.08) +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1),
               linetype = "dashed", color = "grey50", linewidth = 0.7) +
  geom_line(color = "#001965", linewidth = 1) +
  annotate(
    "text",
    x = 0.55,
    y = 0.35,
    label = paste0("AUC = ", sprintf("%.2f", auc_val)),
    size = 7,
    color = "#001965",
    fontface = "bold"
  ) +
  annotate(
    "text",
    x = 0.55,
    y = 0.25,
    label = paste0("Accuracy = ", sprintf("%.1f%%", accuracy * 100)),
    size = 5,
    color = "#001965",
    fontface = "plain"
  ) +
  xlab("False Positive Rate (FPR)") +
  ylab("True Positive Rate (TPR)") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 25),
    axis.text.y = element_text(size = 10, color = "grey60"),
    axis.text.x = element_text(size = 10, color = "grey60"),
    axis.ticks.y = element_line(linewidth = 0.3, color = "grey60"),
    axis.ticks.x = element_line(linewidth = 0.3, color = "grey60"),
    axis.ticks.length = unit(0.15, "cm"),
    legend.position = "none"
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.2),
    limits = c(0, 1),
    expand = c(0.01, 0.01)
  ) +
  scale_y_continuous(
    breaks = seq(0, 1, by = 0.2),
    limits = c(0, 1),
    expand = c(0.01, 0.01)
  ) +
  coord_equal() +
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 0),
               color = "black", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = 0, y = 0, yend = 1),
               color = "black", linewidth = 0.5)

ROCPlot_SNPbag


# --- Save ---
SavePlotROC <- function(filename, plotname, folder = "Plots/") {
  dir.create(folder, showWarnings = FALSE, recursive = TRUE)

  full_path <- paste0(folder, filename)

  ggsave(
    full_path,
    plot = plotname,
    width = 8,
    height = 8,
    dpi = "retina",
    bg = "white"
  )
}

SavePlotROC("ROC_SNPbag.png", ROCPlot_SNPbag)
