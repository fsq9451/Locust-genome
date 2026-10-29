#!/usr/bin/env Rscript
# ============================================================
# 03b_plot_ancestral_CI.R
# 在 24 物种系统发育树(tree24.nwk)上标注祖先状态重建的 95% CI
#
# 输入(与 03_fitContinuous.R 同目录):
#   tree24.nwk
#   ancestral_genome_size_Gb.csv   (log10 Gb 尺度, node/ace/var/lo95/hi95)
#   ancestral_te_percent.csv       (%,        node/ace/var/lo95/hi95)
#   ancestral_chrom_n.csv          (条数,     node/ace/var/lo95/hi95)
# 输出:
#   tree_CI_genome.pdf / tree_CI_te.pdf / tree_CI_chrom.pdf
#
# 依赖: ape, phytools, ggtree, ggplot2
#   ggtree 安装(如未安装):
#     install.packages("BiocManager")
#     BiocManager::install("ggtree")
# 用法: Rscript 03b_plot_ancestral_CI.R
#   默认标注全部内部节点(白底小标签); 若嵌套节点文字重叠,
#   把下方 LAYOUT 改为 "circular"(圆形布局, 节点沿圆周错开)。
#   KEY 关键节点(node 26=Orthoptera, 35=核心 Caelifera)始终红色加粗。
# ============================================================

suppressMessages({
  library(ape)
  library(phytools)
  library(ggtree)
  library(ggplot2)
})

# ---------- 参数 ----------
all_nodes <- TRUE    # TRUE = 标注全部内部节点; FALSE = 仅关键节点
KEY       <- c(26, 35)   # 26 = Orthoptera MRCA, 35 = 核心 Caelifera MRCA(clade_summary_all.csv 验证)
LAYOUT    <- "rectangular"  # 可选 "rectangular" / "circular"(全标重叠时用圆形)
XFRAC     <- 1.45   # 右侧留白倍数(给全部节点 CI 文本留空间, 仅 rectangular 生效)
WIDTH     <- 7.5    # 输出 PDF 宽(英寸)
HEIGHT    <- 9.5    # 输出 PDF 高(英寸)

# ---------- 读入 ----------
tr <- read.tree("tree24.nwk")

read_anc <- function(f) {
  d <- read.csv(f, stringsAsFactors = FALSE)
  d$node <- as.integer(d$node)
  d
}
a_gb  <- read_anc("ancestral_genome_size_Gb.csv")
a_te  <- read_anc("ancestral_te_percent.csv")
a_chr <- read_anc("ancestral_chrom_n.csv")

# genome size: log10(Gb) -> Gb (CI 同步转换)
a_gb[, c("ace", "lo95", "hi95")] <- 10^a_gb[, c("ace", "lo95", "hi95")]

# ---------- 标注文本 ----------
fmt_ci <- function(ace, lo, hi, dec) {
  f <- paste0("%.", dec, "f")
  paste0(sprintf(f, ace), "\n[", sprintf(f, lo), ", ", sprintf(f, hi), "]")
}

# 染色体 CI 含负值(连续性状近似的固有限制)时仅标点估计
if (any(a_chr$lo95 < 0 & !is.na(a_chr$lo95))) {
  neg <- a_chr$lo95 < 0
  a_chr$lab_plain <- sprintf("%.1f", a_chr$ace)
  a_chr$lab_plain[!neg] <- fmt_ci(a_chr$ace[!neg], a_chr$lo95[!neg],
                                  a_chr$hi95[!neg], 1)
  a_chr$lab_full <- a_chr$lab_plain
  message("注意: 染色体数部分节点 95% CI 含负值, 仅标注点估计 (节点 ",
          paste(a_chr$node[neg], collapse = ", "), ")")
} else {
  a_chr$lab_full <- fmt_ci(a_chr$ace, a_chr$lo95, a_chr$hi95, 1)
  a_chr$lab_plain <- a_chr$lab_full
}
a_gb$lab_full <- fmt_ci(a_gb$ace, a_gb$lo95, a_gb$hi95, 2)
a_te$lab_full <- fmt_ci(a_te$ace, a_te$lo95, a_te$hi95, 1)

# ---------- 绘图函数 ----------
plot_ci <- function(d, lab_col, file) {
  p <- ggtree(tr, layout = LAYOUT, size = 0.35) +
    geom_tiplab(size = 2.0, fontface = "italic", offset = 0.012) +
    geom_treescale(x = 0, y = 0, width = 0.1, fontsize = 2.2)
  td <- p$data                       # 树布局坐标
  nd <- td[!td$isTip, c("node", "x", "y")]
  nd <- merge(nd, d, by = "node")
  maxx <- max(td$x, na.rm = TRUE)
  if (LAYOUT == "rectangular") p <- p + xlim(0, maxx * XFRAC)
  if (all_nodes) {
    # 全部内部节点: 白底半透明标签, 避免文字与树枝重叠
    p <- p + geom_label(data = nd, aes(x = x, y = y, label = .data[[lab_col]]),
                        hjust = 0, nudge_x = maxx * 0.006, size = 1.6, color = "grey20",
                        fill = "white", alpha = 0.65, label.size = 0.1,
                        label.padding = unit(0.15, "lines"), lineheight = 0.85)
  }
  # 关键节点: 红色加粗(始终标注)
  p <- p + geom_label(data = subset(nd, node %in% KEY),
                      aes(x = x, y = y, label = .data[[lab_col]]),
                      hjust = 0, nudge_x = maxx * 0.006, size = 2.1,
                      color = "firebrick", fill = "white", alpha = 0.85,
                      fontface = "bold", label.size = 0.1,
                      label.padding = unit(0.15, "lines"), lineheight = 0.85)
  pdf(file, width = WIDTH, height = HEIGHT)
  print(p)
  dev.off()
  invisible(nd)
}

# ---------- 运行三性状 ----------
nd_gb  <- plot_ci(a_gb, "lab_full", "tree_CI_genome.pdf")
nd_te  <- plot_ci(a_te, "lab_full", "tree_CI_te.pdf")
nd_chr <- plot_ci(a_chr, "lab_full", "tree_CI_chrom.pdf")

# ---------- 与正文核对(关键节点) ----------
chk <- function(nd, nm) {
  s <- subset(nd, node %in% KEY)
  cat(sprintf("== %s ==\n", nm))
  for (i in seq_len(nrow(s))) {
    cat(sprintf("  node %d: %s\n", s$node[i], s$lab_full[i]))
  }
}
chk(nd_gb,  "Genome size (Gb)")
chk(nd_te,  "TE content (%)")
chk(nd_chr, "Chromosome number (n)")
cat(">>> 完成: tree_CI_genome.pdf / tree_CI_te.pdf / tree_CI_chrom.pdf\n")


