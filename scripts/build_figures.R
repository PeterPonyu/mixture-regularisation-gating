#!/usr/bin/env Rscript
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
root <- normalizePath(file.path(dirname(script), ".."))
source(file.path(root, "scripts", "plot_helpers.R"))
python <- Sys.getenv("PYTHON", "python3")
render_status <- system2(python, shQuote(c(
  file.path(root, "scripts", "render_schematic.py"),
  file.path(root, "figures", "architecture.svg"),
  file.path(root, "figures", "architecture.pdf"))))
stopifnot(render_status == 0)
capped <- read_sweep("dpmm-transformer-prior-20260907")
uncapped <- read_sweep("dpmm-transformer-prior-unclamped-20260907")
paired_panels(uncapped, "edge", "Edge survival (200 epochs)", "fourbg_unclamped_edge")
quantity_labels <- c(coherence = "Coordinate coherence (v1)", perplexity = "Coordinate perplexity",
  edge = "Edge survival", separation = "Separation survival", reachability = "Reachability survival",
  occupancy = "Effective component count")
heatmap_data <- lapply(list(capped, uncapped), function(data) {
  do.call(rbind, lapply(names(quantity_labels), function(field) {
    do.call(rbind, lapply(as.character(0:2), function(seed_id) {
      pair <- subset(data, dataset == "setty" & axis == "mixture_weight" & value %in% c(0, 1) & seed == seed_id)
      stopifnot(nrow(pair) == 2)
      data.frame(seed = seed_id, quantity = field, delta = abs(diff(pair[order(pair$value), field])))
    }))
  }))
})
historical_deltas <- rbind(transform(heatmap_data[[1]], configuration = 1),
                          transform(heatmap_data[[2]], configuration = 2))
delta_panels <- lapply(names(quantity_labels), function(field) {
  panel <- subset(historical_deltas, quantity == field)
  panel$x <- panel$configuration + (as.numeric(panel$seed) - 1) * 0.08
  ggplot(panel, aes(x, delta, colour = seed, shape = seed)) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.3) +
    geom_point(size = 2.3) + seed_scales() +
    scale_x_continuous(breaks = 1:2, labels = c("Capped", "Uncapped"), limits = c(0.75, 2.25)) +
    scale_y_continuous(limits = c(0, max(0.01, max(panel$delta) * 1.12)),
                       expand = expansion(mult = c(0.06, 0.04))) +
    labs(title = quantity_labels[[field]], x = "Historical configuration",
         y = "Absolute coupling difference")
})
save_figure(delta_panels, "clamp_vs_unclamped", ncol = 3, width = 8.2, height = 4.8,
            point_counts = rep(6, 6))
purity <- read.csv(file.path(root, "evidence", "purity.csv"), stringsAsFactors = FALSE)
purity$seed <- as.character(purity$seed)
purity$coupling <- as.numeric(sub("w=", "", purity$dose))
purity_plots <- function(fields, titles, labels) {
  lapply(seq_along(fields), function(index) {
    purity$score <- purity[[fields[index]]]
    ggplot(purity, aes(coupling, score, colour = seed, shape = seed, group = seed)) +
      geom_line(linewidth = 0.6) + geom_point(size = 2.2) + seed_scales() +
      scale_x_continuous(breaks = c(0, 1), labels = c("w = 0", "w = 1"), limits = c(-0.15, 1.15)) +
      labs(title = titles[index], x = "Uncapped mixture coupling", y = labels[index])
  })
}
plots <- purity_plots("knn15_purity", "Setty UMAP label purity", "First 15 nonself-neighbour purity")
plots[[1]] <- plots[[1]] + scale_y_continuous(limits = c(0, 1))
save_figure(plots, "purity_callout", ncol = 1, width = 6.8, height = 2.6, point_counts = 6)
save_figure(purity_plots(c("mean_radius", "sep_over_rad"),
  c("Within-label radius", "Separation relative to radius"),
  c("Mean within-label radius", "Centroid separation / mean radius")),
  "plane_stats_unclamped", width = 7.8, height = 3.5, point_counts = c(6, 6))

arm_order <- c("w0", "cap5_a100", "cap5_a20", "none_a100", "none_a20")
arm_labels <- c("w = 0", "cap 5\na = 100", "cap 5\na = 20", "no cap\na = 100", "no cap\na = 20")
old_pairs <- read_evidence("factorial_results.json")$pairs
old <- do.call(rbind, lapply(old_pairs, function(pair) {
  data <- rows_frame(pair$runs, c(arm = "arm", edge = "metrics.edge_survival"))
  data$dataset <- pair$dataset
  data$seed <- as.character(pair$seed)
  data
}))
old <- numeric_columns(old, "edge")
new <- rows_frame(read_evidence("new_results.json")$rows,
  c(dataset = "dataset", seed = "seed", arm = "arm", edge = "metrics.edge_survival", dose = "gradient_dose"))
new <- numeric_columns(new, c("edge", "dose"))
stopifnot(nrow(old) == 30, nrow(new) == 30)
factorial_plots <- lapply(c("setty", "endo"), function(background) {
  panel <- subset(old, dataset == background)
  panel$arm <- factor(panel$arm, levels = arm_order)
  ggplot(panel, aes(arm, edge, colour = seed, shape = seed, group = seed)) +
    geom_line(linewidth = 0.6) + geom_point(size = 2) + seed_scales() +
    scale_x_discrete(labels = arm_labels) + scale_y_continuous(limits = c(0.25, 0.8)) +
    labs(title = paste0(background_labels[[background]], ": 20 post-warmup epochs"), x = "Matched arm", y = "Edge survival")
})
save_figure(factorial_plots, "factorial_edge", width = 7.8, height = 3.1, point_counts = c(15, 15))
extension <- list()
for (background in c("setty", "endo")) {
  before <- subset(old, dataset == background & arm %in% c("w0", "none_a100", "none_a20"))
  after <- subset(new, dataset == background & arm %in% c("w0", "none_a100", "none_a20"))
  before$epoch <- 200
  after$epoch <- 220
  paired <- rbind(before, after[, names(before)])
  paired$x <- match(paired$arm, c("w0", "none_a100", "none_a20")) + ifelse(paired$epoch == 200, -0.18, 0.18)
  extension[[length(extension) + 1]] <- ggplot(paired, aes(x, edge, colour = seed, shape = seed, group = interaction(arm, seed))) +
    geom_line(linewidth = 0.5) + geom_point(size = 1.8) + seed_scales() +
    scale_x_continuous(breaks = 1:3, labels = c("w = 0", "no cap\na = 100", "no cap\na = 20")) +
    scale_y_continuous(limits = c(0.25, 0.8)) +
    labs(title = paste0(background_labels[[background]], ": endpoint continuation"),
         x = "Each pair: epoch 200 → 220", y = "Edge survival")
}
for (background in c("setty", "endo")) {
  panel <- subset(new, dataset == background & arm != "w0")
  panel$x <- match(panel$arm, arm_order[-1]) + (as.numeric(panel$seed) - 1) * 0.08
  extension[[length(extension) + 1]] <- ggplot(panel, aes(x, dose, colour = seed, shape = seed)) +
    geom_point(size = 2) + seed_scales() +
    scale_x_continuous(breaks = 1:4, labels = arm_labels[-1]) +
    labs(title = paste0(background_labels[[background]], ": added-window exposure"),
         x = "Matched arm", y = "Sum of weighted\nlatent-gradient norms")
}
save_figure(extension, "window_extension", width = 7.8, height = 5.0, point_counts = c(18, 18, 12, 12))
plot_umap()
write_figure_manifest()
