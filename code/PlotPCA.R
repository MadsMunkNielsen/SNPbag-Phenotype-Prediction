library(ggplot2)

set.seed(42)

# --- Simulation parameters ---
N_per_pop <- 150          # individuals per population
M         <- 5000         # number of SNPs
n_pops    <- 3            # number of ancestral populations
N         <- N_per_pop * n_pops

# --- Simulate ancestral allele frequencies ---
# Base frequencies drawn uniformly, then shifted per population
# to create differentiated allele frequency profiles (mimicking Fst ~ 0.05)
base_freq <- runif(M, 0.1, 0.9)
Fst <- 0.05

# For each population, draw allele frequencies from a Beta distribution
# centered on the base frequency with dispersion governed by Fst
pop_freqs <- matrix(NA, nrow = n_pops, ncol = M)
for (k in 1:n_pops) {
  a <- base_freq * (1 - Fst) / Fst
  b <- (1 - base_freq) * (1 - Fst) / Fst
  pop_freqs[k, ] <- rbeta(M, a, b)
  # Clamp to avoid 0/1
  pop_freqs[k, ] <- pmin(pmax(pop_freqs[k, ], 0.01), 0.99)
}

# --- Simulate genotypes (additive coding: 0, 1, 2) ---
G <- matrix(NA, nrow = N, ncol = M)
pop_labels <- rep(c("Population 1", "Population 2", "Population 3"), each = N_per_pop)

for (k in 1:n_pops) {
  rows <- ((k - 1) * N_per_pop + 1):(k * N_per_pop)
  for (j in 1:M) {
    G[rows, j] <- rbinom(N_per_pop, size = 2, prob = pop_freqs[k, j])
  }
}

# --- PCA on the centred and scaled genotype matrix ---
G_scaled <- scale(G, center = TRUE, scale = TRUE)
# Remove SNPs with zero variance (monomorphic after simulation)
keep <- apply(G_scaled, 2, function(col) !any(is.nan(col)))
G_scaled <- G_scaled[, keep]

pca <- prcomp(G_scaled, center = FALSE, scale. = FALSE)

# Proportion of variance explained
var_explained <- (pca$sdev^2 / sum(pca$sdev^2)) * 100

df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  Population = pop_labels
)

# --- Plot ---
pop_colours <- c(
  "Population 1" = "#001965",
  "Population 2" = "#00857C",
  "Population 3" = "#CB333B"
)

PCAPlot <- ggplot(df, aes(x = PC1, y = PC2, colour = Population)) +
  geom_point(size = 1.4, alpha = 0.7) +
  scale_colour_manual(values = pop_colours) +
  xlab(paste0("PC1 (", round(var_explained[1], 1), "%)")) +
  ylab(paste0("PC2 (", round(var_explained[2], 1), "%)")) +
  theme_minimal() +
  theme(
    axis.title   = element_text(size = 25),
    axis.text.y  = element_text(size = 10, color = "grey60"),
    axis.text.x  = element_text(size = 10, color = "grey60"),
    axis.ticks.y = element_line(size = 0.3, color = "grey60"),
    axis.ticks.x = element_line(size = 0.3, color = "grey60"),
    axis.ticks.length = unit(0.15, "cm"),
    axis.line.x = element_line(linewidth = 0.5, color = "black"),
    axis.line.y = element_line(linewidth = 0.5, color = "black"),
    legend.title = element_text(size = 14),
    legend.text  = element_text(size = 12),
    legend.position = "right"
  ) 

# --- Save ---
SavePlot <- function(filename, plotname, folder = "Plots/") {
  if (!dir.exists(folder)) dir.create(folder, recursive = TRUE)
  full_path <- paste0(folder, filename)
  ggsave(full_path, plot = plotname, bg = "white", width = 13, height = 6, dpi = "retina")
}

SavePlot("PCA_population_structure.png", PCAPlot)



