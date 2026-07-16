## ==========================================================================
##  Script 1_5_DGE_stages — Differential expression across fibrosis stages
##
##  All genes are tested (not only the nuclear mechanics list).
##  Samples: F0 + F1 + F2 + F3 + F4. F0 = NAFLD with steatosis and no
##  fibrosis, used as the baseline. Control_Normal samples are excluded.
##
##  One model per dataset: ~ sex + group, with group = F0..F4.
##  From that single fit two things are taken:
##    - LRT against ~ sex: one p-value per gene for any change across stages.
##    - Wald contrasts: the 10 pairwise comparisons
##        F1/F2/F3/F4 vs F0   (baseline comparisons)
##        F2/F3/F4 vs F1, F3/F4 vs F2, F4 vs F3   (between-strata)
##
##  Outputs (results/dge_stages):
##    dge_contrasts_all.tsv.gz   full statistics, every gene and contrast
##    dge_lrt_stage.tsv          LRT result per gene and dataset
##    dge_consensus_contrasts.tsv  contrasts reproduced across datasets
##    dge_gene_summary.tsv       per-gene summary and class
##    dge_intersections.tsv      set sizes
## ==========================================================================

# paths relative to this script
.file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
BASE  <- if (length(.file)) dirname(normalizePath(.file)) else getwd()
library(DESeq2)
library(dplyr)
library(tidyr)
library(vroom)
library(AnnotationDbi)
library(org.Hs.eg.db)

DIR_OBJ <- file.path(BASE, "geo", "R_objects")
DIR_OUT <- file.path(BASE, "results", "dge_stages")
dir.create(DIR_OUT, recursive = TRUE, showWarnings = FALSE)

## thresholds
PADJ_CUT     <- 0.05
LFC_CUT      <- log2(1.5)
MIN_DATASETS <- 2      # datasets required to call a change reproducible

STAGES   <- c("F0", "F1", "F2", "F3", "F4")
DATASETS <- c("GSE130970", "GSE135251", "GSE162694")

## ---------------------------------------------------------------------------
##  1. Load and subset
## ---------------------------------------------------------------------------

counts <- read.table(
  file.path(DIR_OBJ, "1_1_counts_master.tsv"),
  sep = "\t", header = TRUE, row.names = 1, check.names = FALSE
)

metadata <- read.table(
  file.path(DIR_OBJ, "1_1_metadata_master.tsv"),
  sep = "\t", header = TRUE, stringsAsFactors = FALSE
)

stopifnot(all(colnames(counts) == metadata$sample_id))

keep          <- metadata$histology_group %in% STAGES
metadata_filt <- metadata[keep, ]
counts_filt   <- counts[, keep]

message("Samples kept (F0-F4): ", ncol(counts_filt))
print(table(metadata_filt$dataset, metadata_filt$histology_group))

## ---------------------------------------------------------------------------
##  2. Contrast list: every pair of stages, later stage as numerator
## ---------------------------------------------------------------------------

pairs_mat <- combn(STAGES, 2)
contrasts_tbl <- data.frame(
  numerator   = pairs_mat[2, ],
  denominator = pairs_mat[1, ],
  stringsAsFactors = FALSE
)
contrasts_tbl$contrast <- paste0(contrasts_tbl$numerator, "_vs_",
                                 contrasts_tbl$denominator)
contrasts_tbl$type <- ifelse(contrasts_tbl$denominator == "F0",
                             "baseline", "between_strata")

message("\nContrasts per dataset: ", nrow(contrasts_tbl))

## ---------------------------------------------------------------------------
##  3. Fit one model per dataset and pull LRT + all contrasts
## ---------------------------------------------------------------------------

run_dataset <- function(dataset_name) {

  message("\n=== ", dataset_name, " ===")

  idx     <- metadata_filt$dataset == dataset_name
  meta_ds <- metadata_filt[idx, ]
  mat_ds  <- as.matrix(counts_filt[, idx])
  storage.mode(mat_ds) <- "integer"

  ## drop samples without sex
  ok      <- !is.na(meta_ds$sex)
  if (any(!ok)) message("  Dropping ", sum(!ok), " sample(s) with missing sex")
  meta_ds <- meta_ds[ok, ]
  mat_ds  <- mat_ds[, ok]

  ## same low-count filter used for normalization
  mat_ds  <- mat_ds[rowSums(mat_ds >= 10) >= 3, ]

  coldata <- data.frame(
    row.names = meta_ds$sample_id,
    sex   = factor(meta_ds$sex, levels = c("Male", "Female")),
    group = factor(meta_ds$histology_group, levels = STAGES)
  )

  message("  Samples : ", ncol(mat_ds))
  message("  Genes   : ", nrow(mat_ds))
  print(table(coldata$group))

  dds <- DESeqDataSetFromMatrix(
    countData = mat_ds,
    colData   = coldata,
    design    = ~ sex + group
  )

  ## Wald fit, used for the pairwise contrasts
  dds_wald <- DESeq(dds)

  ## LRT against the model without stage: does the gene change anywhere?
  dds_lrt  <- DESeq(dds, test = "LRT", reduced = ~ sex)

  res_lrt <- as.data.frame(results(dds_lrt)) %>%
    tibble::rownames_to_column("Ensembl_ID") %>%
    transmute(
      dataset      = dataset_name,
      Ensembl_ID,
      baseMean,
      stat_lrt     = stat,
      pvalue_lrt   = pvalue,
      padj_lrt     = padj
    )

  ## every pairwise contrast from the same fit
  res_pairs <- lapply(seq_len(nrow(contrasts_tbl)), function(i) {
    num <- contrasts_tbl$numerator[i]
    den <- contrasts_tbl$denominator[i]
    r   <- results(dds_wald, contrast = c("group", num, den))
    as.data.frame(r) %>%
      tibble::rownames_to_column("Ensembl_ID") %>%
      transmute(
        dataset   = dataset_name,
        contrast  = contrasts_tbl$contrast[i],
        type      = contrasts_tbl$type[i],
        Ensembl_ID,
        baseMean,
        log2FC    = log2FoldChange,
        lfcSE,
        pvalue,
        padj
      )
  }) %>% bind_rows()

  message("  Contrast rows: ", nrow(res_pairs))

  list(lrt = res_lrt, pairs = res_pairs)
}

res_all <- lapply(DATASETS, run_dataset)
names(res_all) <- DATASETS

lrt_long   <- bind_rows(lapply(res_all, `[[`, "lrt"))
pairs_long <- bind_rows(lapply(res_all, `[[`, "pairs"))

## ---------------------------------------------------------------------------
##  4. Gene symbols
## ---------------------------------------------------------------------------

all_ids <- unique(c(lrt_long$Ensembl_ID, pairs_long$Ensembl_ID))
sym_map <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys    = all_ids,
  keytype = "ENSEMBL",
  columns = "SYMBOL"
) %>%
  dplyr::rename(Ensembl_ID = ENSEMBL, Gene_symbol = SYMBOL) %>%
  dplyr::distinct(Ensembl_ID, .keep_all = TRUE)

lrt_long   <- left_join(lrt_long,   sym_map, by = "Ensembl_ID")
pairs_long <- left_join(pairs_long, sym_map, by = "Ensembl_ID")

## ---------------------------------------------------------------------------
##  5. Flag significant results and check reproducibility across datasets
## ---------------------------------------------------------------------------

pairs_long <- pairs_long %>%
  mutate(
    sig = !is.na(padj) & padj < PADJ_CUT & !is.na(log2FC) &
          abs(log2FC) >= LFC_CUT,
    dir = ifelse(!sig, NA_character_, ifelse(log2FC > 0, "up", "down"))
  )

## a contrast is kept when the same direction shows up in enough datasets
consensus_pairs <- pairs_long %>%
  filter(sig) %>%
  group_by(Ensembl_ID, Gene_symbol, contrast, type, dir) %>%
  summarise(
    n_datasets = n_distinct(dataset),
    datasets   = paste(sort(unique(dataset)), collapse = ","),
    mean_log2FC = mean(log2FC),
    max_padj   = max(padj),
    .groups    = "drop"
  ) %>%
  filter(n_datasets >= MIN_DATASETS)

## drop genes where datasets disagree on the direction of the same contrast
conflicting <- consensus_pairs %>%
  count(Ensembl_ID, contrast) %>%
  filter(n > 1) %>%
  dplyr::select(Ensembl_ID, contrast)

consensus_pairs <- anti_join(consensus_pairs, conflicting,
                             by = c("Ensembl_ID", "contrast"))

message("\nConsensus contrast calls: ", nrow(consensus_pairs))

## LRT consensus: gene changes across stages in enough datasets
lrt_consensus <- lrt_long %>%
  mutate(sig_lrt = !is.na(padj_lrt) & padj_lrt < PADJ_CUT) %>%
  group_by(Ensembl_ID, Gene_symbol) %>%
  summarise(
    n_ds_tested  = n_distinct(dataset),
    n_ds_lrt_sig = sum(sig_lrt),
    min_padj_lrt = suppressWarnings(min(padj_lrt, na.rm = TRUE)),
    .groups      = "drop"
  ) %>%
  mutate(
    min_padj_lrt  = ifelse(is.finite(min_padj_lrt), min_padj_lrt, NA_real_),
    lrt_reproduced = n_ds_lrt_sig >= MIN_DATASETS
  )

## ---------------------------------------------------------------------------
##  6. Per-gene summary
##
##  baseline      : gene differs from F0 in at least one stage
##  between_strata: gene differs between two fibrotic stages
##  class         : intersection of both sets
## ---------------------------------------------------------------------------

base_set <- consensus_pairs %>%
  filter(type == "baseline") %>%
  group_by(Ensembl_ID) %>%
  summarise(
    n_baseline_sig   = n(),
    baseline_dirs    = paste(sort(unique(dir)), collapse = "/"),
    first_stage_diff = min(as.integer(sub("^F(\\d)_vs_F0$", "\\1", contrast))),
    baseline_hits    = paste(contrast, collapse = ","),
    .groups = "drop"
  )

strata_set <- consensus_pairs %>%
  filter(type == "between_strata") %>%
  group_by(Ensembl_ID) %>%
  summarise(
    n_strata_sig = n(),
    strata_dirs  = paste(sort(unique(dir)), collapse = "/"),
    strata_hits  = paste(contrast, collapse = ","),
    .groups = "drop"
  )

gene_summary <- lrt_consensus %>%
  left_join(base_set,   by = "Ensembl_ID") %>%
  left_join(strata_set, by = "Ensembl_ID") %>%
  mutate(
    n_baseline_sig = tidyr::replace_na(n_baseline_sig, 0L),
    n_strata_sig   = tidyr::replace_na(n_strata_sig, 0L),
    class = case_when(
      n_baseline_sig > 0 & n_strata_sig > 0 ~ "baseline_and_strata",
      n_baseline_sig > 0                    ~ "baseline_only",
      n_strata_sig   > 0                    ~ "strata_only",
      TRUE                                  ~ "not_significant"
    )
  ) %>%
  arrange(desc(class == "baseline_and_strata"), min_padj_lrt)

## ---------------------------------------------------------------------------
##  7. Set sizes
## ---------------------------------------------------------------------------

intersections <- gene_summary %>%
  summarise(
    genes_tested        = n(),
    lrt_reproduced      = sum(lrt_reproduced),
    baseline_any        = sum(n_baseline_sig > 0),
    strata_any          = sum(n_strata_sig > 0),
    baseline_and_strata = sum(class == "baseline_and_strata"),
    baseline_only       = sum(class == "baseline_only"),
    strata_only         = sum(class == "strata_only"),
    lrt_and_baseline    = sum(lrt_reproduced & n_baseline_sig > 0),
    lrt_and_strata      = sum(lrt_reproduced & n_strata_sig > 0)
  ) %>%
  pivot_longer(everything(), names_to = "set", values_to = "n_genes")

print(as.data.frame(intersections))

## genes per baseline contrast, to see where the response starts
per_contrast <- consensus_pairs %>%
  count(contrast, type, dir) %>%
  pivot_wider(names_from = dir, values_from = n, values_fill = 0)

print(as.data.frame(per_contrast))

## ---------------------------------------------------------------------------
##  8. Write outputs
## ---------------------------------------------------------------------------

vroom_write(pairs_long, file.path(DIR_OUT, "dge_contrasts_all.tsv.gz"))
vroom_write(lrt_long,   file.path(DIR_OUT, "dge_lrt_stage.tsv"))
vroom_write(consensus_pairs, file.path(DIR_OUT, "dge_consensus_contrasts.tsv"))
vroom_write(gene_summary,    file.path(DIR_OUT, "dge_gene_summary.tsv"))
vroom_write(intersections,   file.path(DIR_OUT, "dge_intersections.tsv"))
vroom_write(per_contrast,    file.path(DIR_OUT, "dge_genes_per_contrast.tsv"))

message("\nDone. Files written to ", DIR_OUT)
