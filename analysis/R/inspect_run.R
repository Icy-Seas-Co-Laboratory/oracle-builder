#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: inspect_run.R RUN_DIR")
}

run_dir <- args[[1]]
metadata <- fromJSON(file.path(run_dir, "run_metadata.json"))
print(metadata)

metrics_path <- file.path(run_dir, "metrics.csv")
if (file.exists(metrics_path)) {
  metrics <- read.csv(metrics_path)
  print(tail(metrics))
  if ("loss" %in% names(metrics)) {
    print(ggplot(metrics, aes(epoch, loss)) + geom_line() + theme_minimal())
  }
}

log_path <- file.path(run_dir, "training_log.sqlite")
if (file.exists(log_path)) {
  con <- dbConnect(SQLite(), log_path)
  print(dbGetQuery(con, "SELECT * FROM epoch_metrics ORDER BY epoch, split, metric LIMIT 20"))
  dbDisconnect(con)
}

sample_path <- file.path(run_dir, "evaluation", "sample_metrics.csv")
if (file.exists(sample_path)) {
  print(head(read.csv(sample_path)))
}

