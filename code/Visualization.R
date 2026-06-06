library(ggplot2)

# ── Paths ──────────────────────────────────────────────────────────────────────
analysis_dir <- "SNPbag/checkpoints/analysis"
plots_dir    <- "Plots"

dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

# ── Palette (consistent with PlotGeLu.R / PlotPCA.R / PlotRocAuc.R) ──────────
COLOURS <- c(
  "linear"      = "#001965",
  "interaction" = "#00857C",
  "nonlinear"   = "#CB333B"
)

# ── Shared theme ───────────────────────────────────────────────────────────────
theme_snpbag <- function(...) {
  theme_minimal() +
  theme(
    axis.title        = element_text(size = 25),
    axis.text.y       = element_text(size = 10, color = "grey60"),
    axis.text.x       = element_text(size = 10, color = "grey60"),
    axis.ticks.y      = element_line(linewidth = 0.3, color = "grey60"),
    axis.ticks.x      = element_line(linewidth = 0.3, color = "grey60"),
    axis.ticks.length = unit(0.15, "cm"),
    axis.line.x       = element_line(linewidth = 0.5, color = "black"),
    axis.line.y       = element_line(linewidth = 0.5, color = "black"),
    legend.title      = element_text(size = 14),
    legend.text       = element_text(size = 12),
    legend.position   = "right",
    ...
  )
}

# ── Save helper ────────────────────────────────────────────────────────────────
save_plot <- function(filename, plot, width = 13, height = 6) {
  ggsave(
    file.path(plots_dir, filename),
    plot   = plot,
    width  = width,
    height = height,
    dpi    = "retina",
    bg     = "white"
  )
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Fine-tuning: accuracy by subject size
# ══════════════════════════════════════════════════════════════════════════════
summary_path <- file.path(analysis_dir, "subject_size_summary.csv")

if (file.exists(summary_path)) {
  ft <- read.csv(summary_path)
  ft$phenotype_type <- factor(ft$phenotype_type, levels = names(COLOURS))

  acc_plot <- ggplot(ft, aes(x = subject_size, y = test_accuracy,
                             colour = phenotype_type, group = phenotype_type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.5) +
    scale_colour_manual(values = COLOURS, name = "Phenotype") +
    scale_x_continuous(breaks = unique(ft$subject_size)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    xlab("Number of Individuals") +
    ylab("Test Accuracy") +
    theme_snpbag()

  save_plot("finetune_accuracy_by_subject_size.png", acc_plot)
  message("Saved: finetune_accuracy_by_subject_size.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Fine-tuning: AUC by subject size
# ══════════════════════════════════════════════════════════════════════════════
  auc_plot <- ggplot(ft, aes(x = subject_size, y = test_auc,
                             colour = phenotype_type, group = phenotype_type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.5) +
    scale_colour_manual(values = COLOURS, name = "Phenotype") +
    scale_x_continuous(breaks = unique(ft$subject_size)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    xlab("Number of Individuals") +
    ylab("Test AUC") +
    theme_snpbag()

  save_plot("finetune_auc_by_subject_size.png", auc_plot)
  message("Saved: finetune_auc_by_subject_size.png")
} else {
  message("Skipping fine-tuning summary plots: ", summary_path, " not found.")
}

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Fine-tuning: ROC curves faceted by phenotype type, coloured by N
# ══════════════════════════════════════════════════════════════════════════════
roc_path <- file.path(analysis_dir, "roc_curves.csv")

if (file.exists(roc_path)) {
  roc <- read.csv(roc_path)
  roc$phenotype_type <- factor(roc$phenotype_type, levels = names(COLOURS))
  roc$subject_size   <- factor(roc$subject_size)

  n_sizes   <- nlevels(roc$subject_size)
  size_cols <- colorRampPalette(c("#8fa8d4", "#001965"))(n_sizes)
  names(size_cols) <- levels(roc$subject_size)

  roc_plot <- ggplot(roc, aes(x = fpr, y = tpr,
                              colour = subject_size, group = subject_size)) +
    geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1),
                 linetype = "dashed", color = "grey50", linewidth = 0.6,
                 inherit.aes = FALSE) +
    geom_line(linewidth = 0.9) +
    facet_wrap(~ phenotype_type, ncol = 3) +
    scale_colour_manual(values = size_cols, name = "N individuals") +
    scale_x_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1),
                       expand = c(0.01, 0.01)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1),
                       expand = c(0.01, 0.01)) +
    coord_equal() +
    xlab("False Positive Rate") +
    ylab("True Positive Rate") +
    theme_snpbag(strip.text = element_text(size = 14))

  save_plot("finetune_roc_curves.png", roc_plot, width = 18, height = 6)
  message("Saved: finetune_roc_curves.png")
} else {
  message("Skipping ROC plot: ", roc_path, " not found.")
}

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Pre-training: masked genotype accuracy by subject size
# ══════════════════════════════════════════════════════════════════════════════
pretrain_path <- file.path(analysis_dir, "pretrain_subject_size_summary.csv")

if (file.exists(pretrain_path)) {
  pt <- read.csv(pretrain_path)

  val_acc <- suppressWarnings(as.numeric(pt$best_val_acc))

  pt_long <- data.frame(
    subject_size = c(pt$subject_size, pt$subject_size),
    accuracy     = c(pt$final_train_acc, val_acc),
    split        = c(rep("Train", nrow(pt)), rep("Validation", nrow(pt)))
  )
  pt_long <- pt_long[!is.na(pt_long$accuracy), ]

  pretrain_plot <- ggplot(pt_long, aes(x = subject_size, y = accuracy,
                                       colour = split, group = split)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2.5) +
    scale_colour_manual(
      values = c("Train" = "#001965", "Validation" = "#00857C"),
      name   = "Split"
    ) +
    scale_x_continuous(breaks = unique(pt_long$subject_size)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    xlab("Number of Individuals") +
    ylab("Masked Genotype Accuracy") +
    theme_snpbag()

  save_plot("pretrain_accuracy_by_subject_size.png", pretrain_plot)
  message("Saved: pretrain_accuracy_by_subject_size.png")
} else {
  message("Skipping pre-training plot: ", pretrain_path, " not found.")
}
