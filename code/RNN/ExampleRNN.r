library(ggplot2)
library(ggforce)
library(dplyr)

r <- 0.3  # node radius

# Nodes
nodes <- data.frame(
  id = c("x1","h1","y1",
         "x2","h2","y2",
         "x3","h3","y3"),
  label = c("x[t-2]","","y[t-1]",
            "x[t-1]","","y[t]",
            "x[t]","","y[t+1]"),
  x = c(2,2,2,
        4,4,4,
        6,6,6),
  y = c(0,1,2,   # moved x nodes down to y=0.5
        0,1,2,
        0,1,2)
)


# Edges
edges <- data.frame(
  from = c("x1","h1","h2","x2","x3","h1","h2","h3"),
  to   = c("h1","h2","h3","h2","h3","y1","y2","y3"),
  label = c("W[xh]","h[t-1]","h[t]",
            "W[xh]","W[xh]",
            "W[hy]","W[hy]","W[hy]")
)

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

# Compute edge coordinates
edges_coords <- bind_rows(lapply(1:nrow(edges), function(i){
  row <- edges[i, ]
  from <- nodes[nodes$id == row$from, ]
  to   <- nodes[nodes$id == row$to, ]
  coords <- shorten(from$x, from$y, to$x, to$y, r)
  tibble(
    x = coords[1], y = coords[2],
    xend = coords[3], yend = coords[4],
    label = row$label
  )
}))

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
  label = "h[t+1]"
)

# Separate W and h edges
edges_coords_W <- edges_coords[grepl("^W", edges_coords$label), ]
edges_coords_h <- edges_coords[grepl("^h", edges_coords$label), ]

# Plot
Plot_RNN <- ggplot() +
  geom_circle(data=nodes, aes(x0=x, y0=y, r=r), fill="white", color="black") +
  geom_text(data=nodes, aes(x=x, y=y, label=label), parse=TRUE, size=10) +
  geom_segment(data=edges_coords,
               aes(x=x, y=y, xend=xend, yend=yend),
               arrow=arrow(length=unit(0.2,"inches")), size=1.5) +
  geom_segment(data=extra_edge,
               aes(x=x, y=y, xend=xend, yend=yend),
               arrow=arrow(length=unit(0.2,"inches")), size=1.5) +
  geom_text(data=extra_edge,
            aes(x=(x+xend)/2, y=(y+yend)/2 + 0.2, label=label), # Move label up by 0.2
            parse=TRUE, size=10, vjust=2.4, hjust=0.6) +
  # W edge labels: move up with vjust
  geom_text(data=edges_coords_W,
            aes(x=(x+xend)/2, y=(y+yend)/2, label=label),
            parse=TRUE, size=10, vjust=0.6, hjust=1.3) +
  # h edge labels: move left with hjust
  geom_text(data=edges_coords_h,
            aes(x=(x+xend)/2, y=(y+yend)/2, label=label),
            parse=TRUE, size=10, vjust=1.4, hjust=1) +
  coord_equal() +
  theme_void()

pdf("fig/img/RNN/RNN_Example.pdf", width=10, height=6)
print(Plot_RNN)
dev.off()

getwd()
