# =============================================================
# 02_graft_tree24.R — IQ-TREE 无根树定根 + 6 外群文本嫁接 → 24 物种树
# 输入: tree18.nwk (= iqtree18.treefile 原始输出)
# 嫁接方式: Newick 字符串平级替换(姐妹支长各半), 绕开 ape bind.tip bug
# =============================================================
suppressMessages(library(ape))
suppressMessages(library(phytools))

fix_br <- function(tr) {
  el <- tr$edge.length
  bad <- is.na(el) | el <= 0
  if (any(bad)) {
    ok <- el[!bad]
    el[bad] <- if (length(ok)) min(ok) / 2 else 0.001
    tr$edge.length <- el
    message("修复异常枝长 ", sum(bad), " 条")
  }
  tr
}

tree18 <- read.tree("tree18.nwk")
tree18$node.label <- NULL
if (is.null(tree18$edge.length)) stop("tree18.nwk 无枝长")
tree18 <- root(tree18, outgroup = c("Pame", "Mrel", "Bros", "Pcal", "Tcas"),
               resolve.root = TRUE)
if (!is.binary(tree18)) {
  message("root 后仍非二叉, multi2di 解析")
  tree18 <- multi2di(tree18)
}
tree18 <- fix_br(tree18)
cat("定根后: tips=", Ntip(tree18), " binary=", is.binary(tree18), "\n")

# ---- 文本嫁接 ----
s <- write.tree(tree18)

# ① 平级嫁接: "Sister:len" -> "(Sister:len/2, New:len/2)"
graft_tip <- function(nwk, tip_new, sister) {
  pat  <- paste0(sister, ":([0-9.]+)")
  m    <- regmatches(nwk, regexpr(pat, nwk))
  if (length(m) == 0) stop("找不到 tip: ", sister)
  len  <- as.numeric(sub(".*:", "", m))
  h    <- len / 2
  old  <- paste0(sister, ":", len)
  repl <- paste0("(", sister, ":", h, ",", tip_new, ":", h, ")")
  gsub(old, repl, nwk, fixed = TRUE)
}

s <- graft_tip(s, "Bger", "Pame")
s <- graft_tip(s, "Dtru", "Mrel")
s <- graft_tip(s, "Tchu", "Bros")
s <- graft_tip(s, "Ndub", "Pcal")

# ② Dermaptera ×2: 与 Plecoptera(Pcal,Ndub) 平级姐妹 (2026 Dermoplectopterida)
idx <- regexpr("\\(Pcal:[0-9.]+,Ndub:[0-9.]+\\)", s)
if (idx[1] < 1) stop("找不到 (Pcal,Ndub) 复合节点")
st <- idx[1]; en <- st + attr(idx, "match.length") - 1
m  <- substr(s, st, en)
nd <- as.numeric(sub(".*Ndub:", "", m))
repl <- paste0("(", m, ":", nd, ",(Ctha:", nd, ",Lmin:", nd, "):", nd, ")")
s <- paste0(substr(s, 1, st - 1), repl, substr(s, en + 1, nchar(s)))

# ---- 回读验证 ----
tree24 <- read.tree(text = s)
tree24 <- fix_br(tree24)
stopifnot(is.binary(tree24), Ntip(tree24) == 24)
write.tree(tree24, "tree24.nwk")
cat(">>> tree24.nwk 已生成, tips =", Ntip(tree24), "\n")
cat("拓扑检查: 外群 MRCA =", getMRCA(tree24, c("Pame","Mrel","Bros","Pcal","Tcas")), "\n")
cat("拓扑检查: 直翅目 MRCA =", getMRCA(tree24, c("Asim","Eocu","Gbim","Glon","Lkoh","Lmig","Odec","Oruf","Same","Sgre","Tocc","Vvia","Ztra")), "\n")
cat("拓扑检查: Dermoplectopterida =", getMRCA(tree24, c("Pcal","Ndub","Ctha","Lmin")), "\n")


