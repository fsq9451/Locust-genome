# =============================================================
# 05_TE_stats_landscape.R v7 — TE 汇总 + PGLS + 年轻TE检验 + landscape (聚合版)
# 输入: tbls/*.fa.tbl 与 tbls/*.fa.out, traits.csv, tree24.nwk
# 输出: TE_summary_per_species.csv, PGLS_tree24.txt,
#       TE_landscape_all.pdf, TE_landscape_bins.csv, 年轻TE秩和检验
# =============================================================
suppressMessages({library(ape); library(caper); library(ggplot2); library(dplyr)})

tbl_dir <- "tbls"
tbl_files <- list.files(tbl_dir, pattern = "\\.fa\\.tbl$", full.names = TRUE)
if (!length(tbl_files)) stop("tbls/ 下没有 *.fa.tbl")
species <- sub("\\.fa\\.tbl$", "", basename(tbl_files))
cat("发现物种:", length(species), "\n")

orth13 <- c("Odec","Asim","Eocu","Gbim","Glon","Lkoh","Lmig",
            "Oruf","Same","Sgre","Tocc","Vvia","Ztra")

find_out <- function(sp) {
  p1 <- file.path(tbl_dir, paste0(sp, ".fa.out"))
  if (file.exists(p1)) return(p1)
  p2 <- paste0(sp, "_out/", sp, ".fa.out")
  if (file.exists(p2)) return(p2)
  NA_character_
}

read_out_mat <- function(of) {
  m <- tryCatch(
    if (requireNamespace("data.table", quietly = TRUE))
      as.matrix(data.table::fread(of, header = FALSE, fill = TRUE, skip = 3,
                                  select = c(2L, 6L, 7L), data.table = FALSE))
    else NULL,
    error = function(e) NULL)
  if (is.null(m) || !nrow(m)) {
    ln <- readLines(of, warn = FALSE)
    if (length(ln) <= 3) return(NULL)
    d <- strsplit(ln[-(1:3)], "\\s+")
    d <- d[lengths(d) >= 7]
    if (!length(d)) return(NULL)
    m <- do.call(rbind, lapply(d, function(f) suppressWarnings(as.numeric(f[c(2,6,7)]))))
  }
  m <- matrix(suppressWarnings(as.numeric(m)), ncol = 3)
  m <- m[!is.na(m[,1]) & !is.na(m[,2]) & !is.na(m[,3]), , drop = FALSE]
  if (!nrow(m)) NULL else m
}

n_core <- min(24, parallel::detectCores())
cat("并行核数:", n_core, "\n")
outs <- parallel::mclapply(species, function(sp) {
  of <- find_out(sp)
  if (is.na(of)) return(list(has = FALSE, mat = NULL))
  list(has = TRUE, mat = read_out_mat(of))
}, mc.cores = n_core)
names(outs) <- species
cat("找到 .out:", sum(sapply(outs, `[[`, "has")), "/", length(species), "\n")

read_te_pct <- function(tbl_file) {
  x <- readLines(tbl_file, warn = FALSE)
  ln <- grep("interspersed", x, value = TRUE, ignore.case = TRUE)
  if (!length(ln)) return(NA_real_)
  m <- regexpr("[0-9.]+\\s*%", ln[1])
  if (m[1] > 0) as.numeric(sub("\\s*%$", "", regmatches(ln[1], m)))
  else NA_real_
}
genome_bp <- function(tbl_file, fallback_bp = NA_real_) {
  t <- readLines(tbl_file, warn = FALSE)
  ln <- grep("total length", t, value = TRUE, ignore.case = TRUE)
  if (!length(ln)) { warning(basename(tbl_file), ": 无 total length, 用 traits 兜底"); return(fallback_bp) }
  gb <- as.numeric(gsub("[^0-9]", "", ln[1]))
  if (is.na(gb) || gb > 1e12 || gb < 1e6) {
    warning(basename(tbl_file), ": total length 异常 -> ", ln[1], " 用 traits 兜底"); return(fallback_bp)
  }
  gb
}

dat <- read.csv("traits.csv", stringsAsFactors = FALSE, strip.white = TRUE)
trait_bp <- setNames(dat$genome_size_Gb * 1e9, dat$species)

te_summary <- do.call(rbind, lapply(species, function(sp) {
  tb <- file.path(tbl_dir, paste0(sp, ".fa.tbl"))
  gb <- genome_bp(tb, trait_bp[sp])
  yp <- NA_real_
  if (outs[[sp]]$has && !is.null(outs[[sp]]$mat)) {
    m <- outs[[sp]]$mat
    span <- m[,3] - m[,2] + 1
    yp <- sum(span[m[,1] <= 5]) / gb * 100
  }
  data.frame(species = sp, genome_bp = gb, te_percent = read_te_pct(tb),
             young_percent = yp, stringsAsFactors = FALSE)
}))
te_summary$group <- ifelse(te_summary$species %in% orth13, "Orthoptera", "Outgroup")
write.csv(te_summary, "TE_summary_per_species.csv", row.names = FALSE)
print(te_summary)

run_pgls <- function(tree_file, label) {
  tree <- read.tree(tree_file); tree$node.label <- NULL
  d <- merge(te_summary[, c("species","te_percent")],
             dat[, c("species","genome_size_Gb")], by = "species")
  d <- d[complete.cases(d), ]
  d$gs <- log10(d$genome_size_Gb)
  d <- d[d$species %in% tree$tip.label, ]
  if (nrow(d) < 5) { cat(label, "物种不足, 跳过\n"); return(invisible()) }
  comp <- comparative.data(tree, d, species)
  m <- pgls(gs ~ te_percent, comp)
  cat("\n===== PGLS", label, "(", tree_file, ") n =", nrow(d), "=====\n")
  print(summary(m))
  sink(paste0("PGLS_", label, ".txt")); print(summary(m)); sink()
}
run_pgls("tree24.nwk", "tree24")

wt <- wilcox.test(young_percent ~ group, data = te_summary)
cat("\n年轻TE%(div<=5) 直翅目 vs 外群: W =", wt$statistic, " p =", wt$p.value, "\n")

# ---- landscape: 先按 bin 聚合, 再画图 (避免千万行直绘) ----
land_bins <- do.call(rbind, lapply(species, function(sp) {
  if (!outs[[sp]]$has || is.null(outs[[sp]]$mat)) return(NULL)
  m <- outs[[sp]]$mat
  span <- m[,3] - m[,2] + 1
  gb <- te_summary$genome_bp[te_summary$species == sp]
  tb <- tapply(span, floor(m[,1]), sum)
  data.frame(species = sp, bin = as.integer(names(tb)),
             pct = as.numeric(tb) / gb * 100)
}))
write.csv(land_bins, "TE_landscape_bins.csv", row.names = FALSE)
land <- merge(land_bins, te_summary[, c("species","group")], by = "species")
land <- land[land$bin >= 0 & land$bin <= 50, ]

p <- ggplot(land, aes(x = bin, y = pct, fill = species)) +
  geom_area(position = "stack", alpha = 0.85, linewidth = 0) +
  facet_wrap(~ group, ncol = 1, scales = "free_y") +
  scale_x_continuous(limits = c(0, 50), breaks = seq(0, 50, 10)) +
  labs(x = "Kimura substitution level (%)", y = "Genome fraction (%)",
       title = "TE landscape") +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")
ggsave("TE_landscape_all.pdf", p, width = 8, height = 10)
cat(">>> 已输出 TE_summary / PGLS_tree24.txt / TE_landscape_all.pdf / TE_landscape_bins.csv\n")

