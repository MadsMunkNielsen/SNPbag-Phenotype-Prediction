library(ggplot2)

set.seed(42)

# --- Simulate data ---
n <- 1000

# Covariates
x1 <- rnorm(n, mean = 0, sd = 1)
x2 <- rbinom(n, size = 1, prob = 0.4)
x3 <- rnorm(n, mean = 2, sd = 1.5)

# True linear predictor
eta <- -1.2 + 1.4 * x1 + 0.9 * x2 - 0.7 * x3

# Convert to probabilities using logistic link
p <- 1 / (1 + exp(-eta))

# Binary response
y <- rbinom(n, size = 1, prob = p)

# Put into data frame
dat <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  x3 = x3
)

# Check event rate
mean(dat$y)


# --- Fit GLM ---
fit <- glm(
  y ~ x1 + x2 + x3,
  data = dat,
  family = binomial(link = "logit")
)

summary(fit)

# Predicted probabilities
dat$pred_prob <- predict(fit, type = "response")


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

roc_df <- compute_roc(dat$pred_prob, dat$y)

# Sort before computing AUC
roc_df <- roc_df[order(roc_df$FPR, roc_df$TPR), ]

# --- AUC via trapezoidal rule ---
auc_val <- sum(
  diff(roc_df$FPR) *
    (roc_df$TPR[-1] + roc_df$TPR[-nrow(roc_df)]) / 2
)

auc_val



# --- Plot ROC curve ---
ROCPlot <- ggplot(data = roc_df, aes(x = FPR, y = TPR)) +
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

ROCPlot


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

SavePlotROC("ROC_GLM.png", ROCPlot)


