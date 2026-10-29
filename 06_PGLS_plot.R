# =============================================================
# 06_PGLS_plot.R — log10(genome) ~ TE% 散点 + PGLS 拟合线
# =============================================================
suppressMessages({library(ape); library(caper); library(ggplot2)})
tr <- read.tree("tree24.nwk"); tr$node.label <- NULL
dat <- read.csv("traits.csv", stringsAsFactors = FALSE, strip.white = TRUE)
te  <- read.csv("TE_summary_per_species.csv")
d <- merge(dat[, c("species","genome_size_Gb")],
           te[, c("species","te_percent","group")], by = "species")
d <- d[complete.cases(d), ]
d$gs <- log10(d$genome_size_Gb)
d <- d[d$species %in% tr$tip.label, ]
comp <- comparative.data(tr, d, species)
m <- pgls(gs ~ te_percent, comp)
s  <- summary(m)
slope <- s$coefficients["te_percent", "Estimate"]
intc  <- s$coefficients["(Intercept)", "Estimate"]
pval  <- s$coefficients["te_percent", "Pr(>|t|)"]
cat(sprintf("PGLS: slope=%.4f  p=%.4f  R2=%.3f  n=%d\n", slope, pval, s$r.squared, nrow(d)))

p <- ggplot(d, aes(x = te_percent, y = gs)) +
  geom_point(aes(fill = group), shape = 21, size = 3.2, colour = "grey30") +
  geom_abline(intercept = intc, slope = slope, colour = "#C0392B", linewidth = 0.9) +
  scale_fill_manual(values = c(Orthoptera = "#E67E22", Outgroup = "#5DADE2")) +
  labs(x = "TE content (%)", y = expression(paste(log[10], " genome size (Gb)")),
       fill = NULL,
       caption = sprintf("PGLS: slope = %.4f, p = %.3f, R² = %.3f", slope, pval, s$r.squared)) +
  theme_bw(base_size = 12) +
  theme(legend.position = c(0.15, 0.85))
ggsave("PGLS_genome_vs_TE.pdf", p, width = 6, height = 5)


