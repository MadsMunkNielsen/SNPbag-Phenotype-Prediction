library(ggplot2)

# ── Load data ──────────────────────────────────────────────────────────────────
df_1k  <- read.csv("SNPbag/checkpoints/pretrain/snpbag_n1000_snps14000_seed42_history.csv")
df_5k  <- read.csv("SNPbag/checkpoints/pretrain/snpbag_n5000_snps14000_seed42_history.csv")
df_10k <- read.csv("SNPbag/checkpoints/pretrain/snpbag_n10000_snps14000_seed42_history.csv")
df_50k <- read.csv("SNPbag/checkpoints/pretrain/snpbag_n50000_snps14000_seed42_history.csv")
df_100k <- read.csv("SNPbag/checkpoints/pretrain/snpbag_n100000_snps14000_seed42_history.csv")

df_1k$subjects  <- "1k"
df_5k$subjects  <- "5k"
df_10k$subjects <- "10k"
df_50k$subjects <- "50k"
df_100k$subjects <- "100k"

df <- rbind(df_1k, df_5k, df_10k, df_50k, df_100k)

# Factor ordering so legend reads small → large
df$subjects <- factor(df$subjects, levels = c("1k", "5k", "10k", "50k", "100k"))

# ── Novo Nordisk corporate colour palette ─────────────────────────────────────
novo_colours <- c(
  "1k"   = "#001965",   # True Blue
  "5k"   = "#0070C0",   # Sea Blue
  "10k"  = "#00857C",   # Ocean Green
  "50k"  = "#E8A100",   # Lava Red
  "100k" = "#CB333B"    # Golden Yellow
)

# ── Plot ───────────────────────────────────────────────────────────────────────
ValLossPlot <- ggplot(df, aes(x = epoch, y = val_loss, color = subjects)) +
  geom_line(size = 1) +
  xlab("Epoch") +
  ylab("Validation Loss") +
  theme_minimal() +
  theme(
    axis.title       = element_text(size = 25),
    axis.text.y      = element_text(size = 10, color = "grey60"),
    axis.text.x      = element_text(size = 10, color = "grey60"),
    axis.ticks.y     = element_line(size = 0.3, color = "grey60"),
    axis.ticks.x     = element_line(size = 0.3, color = "grey60"),
    axis.ticks.length = unit(0.15, "cm"),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    legend.title     = element_text(size = 14),
    legend.text      = element_text(size = 12),
    legend.position  = "right"
  ) +
  scale_color_manual(
    name   = "Subjects",
    values = novo_colours,
    drop   = FALSE          # keeps 50k and 100k in the legend even without data
  ) +
  scale_x_continuous(breaks = seq(0, 20, by = 5)) +
  scale_y_continuous() +
  geom_segment(aes(x = 0, xend = 20, y = 0, yend = 0),
               color = "black", linewidth = 0.5) +
  geom_segment(aes(x = 0, xend = 0, y = 0, yend = 0.5),
               color = "black", linewidth = 0.5)

# ── Save ───────────────────────────────────────────────────────────────────────
SavePlot <- function(filename, plotname, folder = "Plots/") {
  full_path <- paste0(folder, filename)
  ggsave(full_path, plot = plotname, width = 13, height = 6, dpi = "retina")
}

SavePlot("ValLoss.png", ValLossPlot)