#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: compare_runs.R RUN_DIR [RUN_DIR ...]")
}

rows <- lapply(args, function(run_dir) {
  metadata <- fromJSON(file.path(run_dir, "run_metadata.json"))
  summary_path <- file.path(run_dir, "evaluation", "evaluation_summary.json")
  summary <- if (file.exists(summary_path)) fromJSON(summary_path) else list()
  as.data.frame(c(list(run = basename(run_dir), status = metadata$status), summary), stringsAsFactors = FALSE)
})

print(do.call(rbind, rows))

