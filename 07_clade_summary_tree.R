# =============================================================
# 07_clade_summary_tree.R — 每个冠群(内部节点)的性状汇总 + 树图
# 输入: tree24.nwk, traits.csv(含 te_percent)
# 输出: clade_summary_all.csv, tree_clade_genome.pdf / te / chrom
# 说明: 冠群统计 = 该节点全部后代物种的实测值(均值±SD, 范围)
# =============================================================
suppressMessages({library(ape)})

tr <- read.tree("tree24.nwk")
tr$node.label <- NULL
dat <- read.csv("traits.csv", stringsAsFactors = FALSE, strip.white = TRUE)
rownames(dat) <- dat$species

summ <- function(v) {
  v <- v[!is.na(v)]; n <- length(v)
  if (!n) return(c(n = 0, mean = NA, sd = NA, min = NA, max = NA))
  if (n == 1) c(n = n, mean = v, sd = 0, min = v, max = v)
  else c(n = n, mean = mean(v), sd = sd(v), min = min(v), max = max(v))
}

N <- Ntip(tr)
nodes <- (N + 1):(N + Nnode(tr))
res <- do.call(rbind, lapply(nodes, function(nd) {
  tips <- extract.clade(tr, nd)$tip.label
  g <- summ(dat[tips, "genome_size_Gb"])
  t <- summ(dat[tips, "te_percent"])
  c <- summ(dat[tips, "chrom_n"])
  data.frame(node = nd, n_tips = g["n"], tips = paste(tips, collapse = ","),
             genome_mean = g["mean"], genome_sd = g["sd"],
             genome_range = sprintf("%.2f-%.2f", g["min"], g["max"]),
             te_mean = t["mean"], te_sd = t["sd"],
             te_range = sprintf("%.1f-%.1f", t["min"], t["max"]),
             chrom_mean = c["mean"], chrom_sd = c["sd"],
             chrom_range = sprintf("%.0f-%.0f", c["min"], c["max"]))
}))
write.csv(res, "clade_summary_all.csv", row.names = FALSE)
print(res[, c("node","n_tips","genome_mean","genome_range",
              "te_mean","te_range","chrom_mean","chrom_range")])

for (lab in c("genome", "te", "chrom")) {
  if (lab == "genome") { vals <- res$genome_mean; main <- "Clade mean genome size (Gb)" }
  if (lab == "te")     { vals <- res$te_mean;     main <- "Clade mean TE content (%)" }
  if (lab == "chrom")  { vals <- res$chrom_mean;  main <- "Clade mean chromosome number" }
  pdf(paste0("tree_clade_", lab, ".pdf"), width = 9, height = 12)
  plot(tr, show.tip.label = TRUE, cex = 0.8, edge.width = 1.5,
       label.offset = 0.02, main = main)
  nodelabels(round(vals, 2), node = nodes, adj = c(1.1, -0.4),
             frame = "none", cex = 0.55, col = "#7B241C")
  add.scale.bar()
  dev.off()
}
cat(">>> 已输出 clade_summary_all.csv / tree_clade_genome.pdf / tree_clade_te.pdf / tree_clade_chrom.pdf\n")


