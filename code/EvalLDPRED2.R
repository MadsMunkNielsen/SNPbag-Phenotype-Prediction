library(ggplot2)

# ============================================================
# ROC Curve, AUC, and Accuracy for LDpred2 PRS GLM
# ============================================================

# --- Load LDpred2 predictions and fit GLM ---
x <- read.csv("SNPbag/TestData/ResultHapPheno/ldpred2_predictions.csv")
pred <- glm(phenotype ~ prs, family = binomial(), data = x)
summary(pred)

# Predicted probabilities
x$pred_prob <- predict(pred, type = "response")


# --- Compute ROC curve manually ---
compute_roc <- function(scores, labels) {
  thresholds <- sort(unique(scores), decreasing = TRUE)

  tpr <- numeric(length(thresholds))
  fpr <- numeric(length(thresholds))

  P <- sum(labels == 1)
  N <- sum(labels == 0)

  for (i in seq_along(thresholds)) {
    pred_pos <- scores >= thresholds[i]

    tp <- sum(pred_pos & labels == 1)
    fp <- sum(pred_pos & labels == 0)

    tpr[i] <- tp / P
    fpr[i] <- fp / N
  }

  data.frame(
    FPR = c(0, fpr, 1),
    TPR = c(0, tpr, 1)
  )
}

roc_df <- compute_roc(x$pred_prob, x$phenotype)

# Sort before computing AUC
roc_df <- roc_df[order(roc_df$FPR, roc_df$TPR), ]


# --- AUC via trapezoidal rule ---
auc_val <- sum(
  diff(roc_df$FPR) *
    (roc_df$TPR[-1] + roc_df$TPR[-nrow(roc_df)]) / 2
)

cat("AUC:", round(auc_val, 4), "\n")


# --- Accuracy at optimal threshold (Youden's J) ---
# Find the threshold that maximises TPR - FPR
roc_inner <- roc_df[-c(1, nrow(roc_df)), ]
thresholds <- sort(unique(x$pred_prob), decreasing = TRUE)
youden_j  <- roc_inner$TPR - roc_inner$FPR
best_idx  <- which.max(youden_j)
best_thresh <- thresholds[best_idx]

# Classify at optimal threshold
x$pred_class <- ifelse(x$pred_prob >= best_thresh, 1, 0)

# Confusion matrix components
tp <- sum(x$pred_class == 1 & x$phenotype == 1)
tn <- sum(x$pred_class == 0 & x$phenotype == 0)
fp <- sum(x$pred_class == 1 & x$phenotype == 0)
fn <- sum(x$pred_class == 0 & x$phenotype == 1)

accuracy   <- (tp + tn) / nrow(x)
sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)

cat("Optimal threshold:", round(best_thresh, 4), "\n")
cat("Accuracy:        ", round(accuracy, 4), "\n")
cat("Sensitivity:     ", round(sensitivity, 4), "\n")
cat("Specificity:     ", round(specificity, 4), "\n")


# --- Plot ROC curve ---
ROCPlot_LDpred2 <- ggplot(data = roc_df, aes(x = FPR, y = TPR)) +
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

ROCPlot_LDpred2


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

SavePlotROC("ROC_GLM_LDpred2.png", ROCPlot_LDpred2)