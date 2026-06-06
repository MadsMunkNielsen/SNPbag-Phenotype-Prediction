library(ggplot2)

# Generate data
x <- seq(-4, 4, length.out = 1000)
gelu <- 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

df <- data.frame(x = x, GELU = gelu)

# Plot
GELUPlot <- ggplot(data = df, aes(x = x)) +
  geom_line(aes(y = GELU), color = "#001965", size = 1) +
  ylab("GELU(x)") +
  xlab("x") +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 25),
    axis.text.y = element_text(size = 10, color = "grey60"),
    axis.text.x = element_text(size = 10, color = "grey60"),
    axis.ticks.y = element_line(size = 0.3, color = "grey60"),
    axis.ticks.x = element_line(size = 0.3, color = "grey60"),
    axis.ticks.length = unit(0.15, "cm"),
    legend.position = "none"
  ) +
  scale_x_continuous(breaks = seq(-4, 4, by = 1)) +
  scale_y_continuous(breaks = seq(-1, 4, by = 1)) +
  geom_segment(aes(x = min(x), xend = max(x), y = 0, yend = 0), color = "black", size = 0.5) +
  geom_segment(aes(x = 0, xend = 0, y = min(GELU) - 0.2, yend = max(GELU) + 0.2), color = "black", size = 0.5)

# Save
SavePlotGELU <- function(filename, plotname, folder = "Plots/") {
  full_path <- paste0(folder, filename)
  ggsave(full_path, plot = plotname, width = 13, height = 6, dpi = "retina")
}

SavePlotGELU("GELU.png", GELUPlot)