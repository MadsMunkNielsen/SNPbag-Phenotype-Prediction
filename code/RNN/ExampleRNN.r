library(ggplot2)
library(ggforce)
library(dplyr)

r <- 0.3  # node radius

# Nodes
nodes <- data.frame(
  id = c("x1","h1","y1",
         "x2","h2","y2",
         "x3","h3","y3"),
  label = c("x[n-1]","h[n-1]","hat(y)[n-1]",
            "x[n]","h[n]","hat(y)[n]",
            "x[n+1]","h[n+1]","hat(y)[n+1]"),
  x = c(2,2,2,
        4,4,4,
        6,6,6),
  y = c(0,1,2,
        0,1,2,
        0,1,2)
)


# Edges
edges <- data.frame(
  from = c("x1","h1","h2","x2","x3","h1","h2","h3"),
  to   = c("h1","h2","h3","h2","h3","y1","y2","y3"),
  label = c("U","W","W",
            "U","U",
            "V","V","V")
)

# Only use the original edges (no yback arrow)
edges_all <- edges

# Function to shorten arrows
shorten <- function(x1, y1, x2, y2, r){
  dx <- x2 - x1
  dy <- y2 - y1
  d <- sqrt(dx^2 + dy^2)
  if (d == 0) return(c(x1, y1, x2, y2))
  x1s <- x1 + dx * r/d
  y1s <- y1 + dy * r/d
  x2s <- x2 - dx * r/d
  y2s <- y2 - dy * r/d
  c(x1s, y1s, x2s, y2s)
}

# Compute edge coordinates for all edges
edges_coords_all <- bind_rows(lapply(1:nrow(edges_all), function(i){
  row <- edges_all[i, ]
  from <- nodes[nodes$id == row$from, ]
  to   <- nodes[nodes$id == row$to, ]
  coords <- shorten(from$x, from$y, to$x, to$y, r)
  tibble(
    x = coords[1], y = coords[2],
    xend = coords[3], yend = coords[4],
    label = row$label
  )
}))

# Offset reverse edges higher but keep them connected to nodes
rev_idx <- edges_coords_all$label == "h[rev]"
edges_coords_all$y[rev_idx]    <- edges_coords_all$y[rev_idx]    + 0
edges_coords_all$yend[rev_idx] <- edges_coords_all$yend[rev_idx] + 0

# If you want to move the original h edges down as well:
orig_h_idx <- edges_coords_all$label == "h[t-1]" | edges_coords_all$label == "h[t]"
edges_coords_all$y[orig_h_idx]    <- edges_coords_all$y[orig_h_idx]    + 0.1
edges_coords_all$yend[orig_h_idx] <- edges_coords_all$yend[orig_h_idx] + 0.1

# Offset yback edge to the right but keep it connected to nodes
yback_idx <- edges_coords_all$label == "yback"
edges_coords_all$x[yback_idx]    <- edges_coords_all$x[yback_idx]    + 0.1
edges_coords_all$xend[yback_idx] <- edges_coords_all$xend[yback_idx] + 0.1

# Calculate offset for extra edge so it starts at the edge of the node
dx <- 1  # x direction (to the right)
dy <- 0  # y direction (horizontal)
d <- sqrt(dx^2 + dy^2)
x_start <- nodes$x[nodes$id == "h3"] + dx * r / d
y_start <- nodes$y[nodes$id == "h3"] + dy * r / d
x_end <- nodes$x[nodes$id == "h3"] + 1
y_end <- nodes$y[nodes$id == "h3"]

extra_edge <- data.frame(
  x = x_start,
  y = y_start,
  xend = x_end,
  yend = y_end,
  label = "W"
)

# Use the same length as the h[t+1] edge for double edges
edge_length <- x_end - nodes$x[nodes$id == "h3"] # This is 1

# Make double edges longer by increasing edge_length
long_edge_length <- edge_length * 1.5

x_left_long <- nodes$x[nodes$id == "h1"] - long_edge_length
y_left <- nodes$y[nodes$id == "h1"]

# Edge going into h1 (above, W[hh] label)
coords_in <- shorten(x_left_long, y_left + 0, nodes$x[nodes$id == "h1"], nodes$y[nodes$id == "h1"] + 0, r)
edge_in <- tibble(
  x = coords_in[1], y = coords_in[2],
  xend = coords_in[3], yend = coords_in[4],
  label = "W"
)

# Only keep the edge_in for plotting, not edge_out
double_edges_split <- bind_rows(edge_in)
double_edges_split$label_offset <- c(-0.18) # Place label under the edge

# Add a label_offset column for vertical adjustment
double_edges_split$label_offset <- c(-0.18) # Place label under the edge

# Separate W and h edges
edges_coords_W <- edges_coords_all[grepl("^W", edges_coords_all$label), ]
edges_coords_h <- edges_coords_all[grepl("^h", edges_coords_all$label), ]

edges_coords_U <- edges_coords_all[edges_coords_all$label == "U", ]
edges_coords_V <- edges_coords_all[edges_coords_all$label == "V", ]


# Plot
Plot_RNN <- ggplot() +
  geom_circle(data=nodes, aes(x0=x, y0=y, r=r), fill="white", color="black") +
  geom_text(data=nodes, aes(x=x, y=y, label=label), parse=TRUE, size=8) +
  geom_segment(data=edges_coords_all,
               aes(x=x, y=y, xend=xend, yend=yend),
               arrow=arrow(length=unit(0.15,"inches")), size=1) +
  geom_segment(data=extra_edge,
               aes(x=x, y=y, xend=xend, yend=yend),
               arrow=arrow(length=unit(0.15,"inches")), size=1) +
  geom_segment(data=double_edges_split,
               aes(x=x, y=y, xend=xend, yend=yend),
               arrow=arrow(length=unit(0.15,"inches")), size=1) +
  geom_text(data=extra_edge,
            aes(x=(x+xend)/2, y=(y+yend)/2 + 0.2, label=label),
            parse=TRUE, size=6, vjust=3.8, hjust=0.6) +
  geom_text(data=edges_coords_W[edges_coords_W$label != "W[hh]", ],
            aes(x=(x+xend)/2, y=(y+yend)/2, label=label),
            parse=TRUE, size=6, vjust=2, hjust=1.3) +
  geom_text(data=edges_coords_h,
            aes(x=(x+xend)/2, y=(y+yend)/2, label=label),
            parse=TRUE, size=6, vjust=-0.5, hjust=0.5) +
  geom_text(data=double_edges_split,
            aes(x=(x+xend)/2, y=(y+yend)/2 + label_offset, label=label),
            parse=TRUE, size=6, vjust=0.6, hjust=0.5) +
  geom_text(data=edges_coords_U,
          aes(x=(x+xend)/2, y=(y+yend)/2, label=label),
          parse=TRUE, size=6, vjust=0.7, hjust=1.8) +
  geom_text(data=edges_coords_V,
          aes(x=(x+xend)/2, y=(y+yend)/2, label=label),
          parse=TRUE, size=6, vjust=0.7, hjust=2) +
  coord_equal() +
  theme_void()

pdf("fig/img/RNN/RNN_Example.pdf", width=10, height=6)
Plot_RNN
dev.off()