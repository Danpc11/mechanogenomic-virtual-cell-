## ==========================================================================
##  plots — per-gene trajectory plots, split up/down
##  Genes ordered by systematic change across the three datasets.
##
##  Input : concordant 3/3 genes from 1_3
##  Output: results/plots/
##    systematic_ranking.tsv
##    trajectories_per_gene_up_systematic.pdf/png
##    trajectories_per_gene_down_systematic.pdf/png
## ==========================================================================

# paths relative to this script
.file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
BASE  <- if (length(.file)) dirname(normalizePath(.file)) else getwd()
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(scales)
library(vroom)

DIR_IN   <- file.path(BASE, "results", "stage_means")
DIR_FITS <- file.path(BASE, "results", "model_fits")
DIR_OUT  <- file.path(BASE, "results", "plots")
dir.create(DIR_OUT, recursive = TRUE, showWarnings = FALSE)

## ---------------------------------------------------------------------------
##  1. Load data and normalize F1-F4 against F0
## ---------------------------------------------------------------------------

fits <- vroom(file.path(DIR_FITS, "model_fits.tsv"),
              show_col_types = FALSE) %>% as.data.frame()

conserved <- vroom(file.path(DIR_FITS, "conserved_genes.tsv"),
                   show_col_types = FALSE) %>% as.data.frame()

df_mean <- vroom(file.path(DIR_IN, "mean_expr_per_stage.tsv"),
                 show_col_types = FALSE) %>% as.data.frame()

f0_ref <- df_mean %>%
  dplyr::filter(stage == 0, !is.na(mean_expr), mean_expr > 0) %>%
  dplyr::select(dataset, Gene_symbol, Ensembl_ID, f0 = mean_expr)

df_norm <- df_mean %>%
  dplyr::filter(stage %in% c(1, 2, 3, 4)) %>%
  dplyr::inner_join(f0_ref, by = c("dataset", "Gene_symbol", "Ensembl_ID")) %>%
  dplyr::mutate(norm_expr = (mean_expr - f0) / f0) %>%
  dplyr::select(-f0)

## ---------------------------------------------------------------------------
##  2. Systematic change score
## ---------------------------------------------------------------------------

## -- 2a. mean absolute delta and CV across datasets -------------------------
#  CV_delta = sd/mean of the three deltas; low CV means the datasets agree.
delta_agg <- conserved %>%
  rowwise() %>%
  mutate(
    d1             = abs(GSE130970_delta_F4_F1),
    d2             = abs(GSE135251_delta_F4_F1),
    d3             = abs(GSE162694_delta_F4_F1),
    mean_abs_delta = mean(c(d1, d2, d3)),
    cv_delta       = sd(c(d1, d2, d3)) / mean_abs_delta
  ) %>%
  ungroup() %>%
  select(Gene_symbol, Ensembl_ID, dominant_dir, mean_abs_delta, cv_delta)

## -- 2b. monotonicity: fraction of consecutive steps (F0-F4) going the
##        right way, averaged over the three datasets ----------------------
mono_raw <- df_mean %>%
  inner_join(
    select(conserved, Gene_symbol, Ensembl_ID, dominant_dir),
    by = c("Gene_symbol", "Ensembl_ID")
  ) %>%
  filter(histology_group != "Control_Normal") %>%
  arrange(Gene_symbol, Ensembl_ID, dataset, stage) %>%
  group_by(Gene_symbol, Ensembl_ID, dominant_dir, dataset) %>%
  summarise(expr_vals = list(mean_expr), .groups = "drop")

mono_raw$diffs     <- lapply(mono_raw$expr_vals, diff)
mono_raw$n_steps   <- sapply(mono_raw$diffs, length)
mono_raw$n_correct <- mapply(
  function(d, dir) if (dir == "up") sum(d > 0) else sum(d < 0),
  mono_raw$diffs, mono_raw$dominant_dir
)
mono_raw$mono_frac <- mono_raw$n_correct / mono_raw$n_steps

mono_agg <- mono_raw %>%
  group_by(Gene_symbol, Ensembl_ID) %>%
  summarise(mean_mono = mean(mono_frac), .groups = "drop")

## -- 2c. combined score and ranking ---------------------------------------
#  systematic_score = mean_abs_delta / (1 + cv_delta)
#    mean_abs_delta : average F1-F4 change (primary)
#    1/(1+cv_delta) : agreement across datasets (secondary)
ranking <- delta_agg %>%
  left_join(mono_agg, by = c("Gene_symbol", "Ensembl_ID")) %>%
  mutate(systematic_score = mean_abs_delta / (1 + cv_delta)) %>%
  arrange(desc(systematic_score))

message("\n--- Top 15 genes by systematic change ---")
print(ranking %>%
  select(Gene_symbol, dominant_dir, mean_abs_delta, cv_delta, systematic_score) %>%
  head(15))

write.table(
  ranking,
  file.path(DIR_OUT, "systematic_ranking.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE
)
message("systematic_ranking.tsv saved")

## ---------------------------------------------------------------------------
##  3. Conserved genes joined with score
## ---------------------------------------------------------------------------

genes_conserved <- conserved %>%
  select(Gene_symbol, Ensembl_ID, dominant_dir) %>%
  left_join(
    select(ranking, Gene_symbol, Ensembl_ID, systematic_score),
    by = c("Gene_symbol", "Ensembl_ID")
  )

fits_conserved <- fits %>%
  inner_join(genes_conserved, by = c("Gene_symbol", "Ensembl_ID"))

norms_conserved <- df_norm %>%
  inner_join(genes_conserved, by = c("Gene_symbol", "Ensembl_ID"))

## ---------------------------------------------------------------------------
##  4. Predicted curves in normalized space (x = 1 to 4)
## ---------------------------------------------------------------------------

x_seq <- seq(1, 4, by = 0.05)

pred_curves <- fits_conserved %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    x_pred = list(x_seq),
    y_pred = list(
      if      (model == "linear")    coef_a + coef_b * x_seq
      else if (model == "power_law") coef_a * x_seq^coef_b
      else if (model == "sigmoid")   coef_a / (1 + exp(-coef_b * (x_seq - coef_c)))
    )
  ) %>%
  dplyr::ungroup() %>%
  tidyr::unnest(c(x_pred, y_pred))

## ---------------------------------------------------------------------------
##  5. Palettes and theme
## ---------------------------------------------------------------------------

dataset_colors <- c(
  "GSE130970" = "#E63946",
  "GSE135251" = "#457B9D",
  "GSE162694" = "#2A9D8F"
)

model_lty <- c(
  "linear"    = "solid",
  "power_law" = "longdash",
  "sigmoid"   = "dotted"
)

theme_pub <- theme_minimal(base_size = 20) +
  theme(
    strip.background  = element_rect(fill = "#1D3557", color = NA),
    strip.text        = element_text(color = "white", face = "bold", size = 18),
    strip.clip        = "off",
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(color = "grey90", linewidth = 0.4),
    panel.border      = element_rect(color = "grey70", fill = NA, linewidth = 0.5),
    legend.position   = "bottom",
    legend.text       = element_text(size = 17),
    legend.title      = element_text(size = 18, face = "bold"),
    axis.text         = element_text(size = 16),
    axis.title        = element_text(size = 18, face = "bold"),
    plot.title        = element_text(face = "bold", size = 22),
    plot.subtitle     = element_text(size = 15, color = "grey40"),
    plot.margin       = margin(14, 20, 14, 14)
  )

## ---------------------------------------------------------------------------
##  6. Gene order by systematic score
## ---------------------------------------------------------------------------

gene_order <- ranking %>%
  arrange(desc(systematic_score)) %>%
  dplyr::distinct(Gene_symbol, dominant_dir)

pred_curves$Gene_symbol <- factor(pred_curves$Gene_symbol,
                                  levels = gene_order$Gene_symbol)

norms_pg <- norms_conserved %>%
  dplyr::filter(!is.na(norm_expr)) %>%
  dplyr::mutate(Gene_symbol = factor(Gene_symbol, levels = gene_order$Gene_symbol))

## ---------------------------------------------------------------------------
##  7. Per-gene plots split by direction (up / down)
## ---------------------------------------------------------------------------

for (dir_split in c("up", "down")) {

  genes_dir <- gene_order %>% dplyr::filter(dominant_dir == dir_split)
  if (nrow(genes_dir) == 0) next
  n_dir <- nrow(genes_dir)

  curves_dir <- pred_curves %>%
    dplyr::filter(dominant_dir == dir_split, !is.na(y_pred)) %>%
    dplyr::mutate(Gene_symbol = factor(Gene_symbol, levels = genes_dir$Gene_symbol))

  norms_dir <- norms_pg %>%
    dplyr::filter(dominant_dir == dir_split) %>%
    dplyr::mutate(Gene_symbol = factor(Gene_symbol, levels = genes_dir$Gene_symbol))

  ncol_dir   <- if (dir_split == "up") 8 else 6
  nrow_dir   <- ceiling(n_dir / ncol_dir)
  width_dir  <- ncol_dir * 7.5
  height_dir <- 5.0 + nrow_dir * 9.0

  p_dir <- ggplot() +

    geom_hline(yintercept = 0, linetype = "dashed",
               color = "grey55", linewidth = 0.7) +

    geom_line(
      data    = curves_dir,
      mapping = aes(x        = x_pred,
                    y        = y_pred,
                    color    = dataset,
                    linetype = model,
                    group    = interaction(dataset, Gene_symbol, Ensembl_ID)),
      linewidth = 2.8, alpha = 0.90
    ) +

    geom_point(
      data    = norms_dir,
      mapping = aes(x = stage, y = norm_expr, color = dataset),
      size = 5.0, alpha = 0.85, show.legend = FALSE
    ) +

    scale_color_manual(values = dataset_colors, name = "Dataset") +
    scale_linetype_manual(values = model_lty, name = "Model") +
    scale_x_continuous(
      breaks = c(1, 2, 3, 4),
      labels = c("F1", "F2", "F3", "F4")
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1)
    ) +

    facet_wrap(~ Gene_symbol, ncol = ncol_dir, scales = "free_y") +

    labs(
      title = if (dir_split == "up")
        "Up-regulated Nuclear Mechanosensitive Genes in Hepatic Fibrosis"
      else
        "Down-regulated Nuclear Mechanosensitive Genes in Hepatic Fibrosis",
      x = "Fibrosis stage",
      y = "Relative change from F0 [(FX - F0)/F0]"
    ) +

    theme_pub +
    theme(
      strip.text       = element_text(size = 52, face = "bold"),
      axis.text        = element_text(size = 34),
      axis.text.x      = element_text(size = 44, face = "bold"),
      axis.title       = element_text(size = 40, face = "bold"),
      plot.title       = element_text(size = 60, face = "bold"),
      plot.subtitle    = element_blank(),
      legend.text      = element_text(size = 52, face = "bold"),
      legend.title     = element_text(size = 56, face = "bold"),
      legend.key.size  = unit(3.0, "cm"),
      legend.key.width = unit(6.5, "cm"),
      legend.spacing.x = unit(2.0, "cm"),
      panel.spacing    = unit(3.0, "lines")
    ) +
    guides(
      color    = guide_legend(override.aes = list(linewidth = 4.5),
                              nrow = 1, byrow = TRUE),
      linetype = guide_legend(override.aes = list(linewidth = 3.5),
                              keywidth = unit(6.5, "cm"), nrow = 1, byrow = TRUE)
    )

  out_name <- file.path(DIR_OUT,
                        paste0("trajectories_per_gene_", dir_split, "_systematic"))
  ggsave(paste0(out_name, ".pdf"), plot = p_dir,
         width = width_dir, height = height_dir, device = pdf, limitsize = FALSE)
  ggsave(paste0(out_name, ".png"), plot = p_dir,
         width = width_dir, height = height_dir, dpi = 220, limitsize = FALSE)
  message("per_gene_", dir_split, " — ", nrow_dir, " rows x ", ncol_dir,
          " cols  |  ", width_dir, " x ", round(height_dir, 1), " in")
}

message("\n=== plots done ===")
message("Outputs -> ", DIR_OUT)
