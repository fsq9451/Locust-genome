#http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/112-pca-principal-component-analysis-essentials/
setwd("doc/PCA")
library("FactoMineR")
library("ggplot2")
library("factoextra")
library("corrplot")
data(decathlon2)

data <- read.table("data.txt", header = TRUE,row.names = 1)
res.pca <- PCA(data[,-8], graph = FALSE)
print(res.pca)

eig.val <- get_eigenvalue(res.pca)
eig.val
percentage <- fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 53))
ggsave("dim_percentage.tiff", plot = percentage, width =130, height = 130,
       units="mm", dpi=300)


var <- get_pca_var(res.pca)
var
# Coordinates
head(var$coord)
# Cos2: quality on the factore map
head(var$cos2)
# Contributions to the principal components
contri <- head(var$contrib)
write.csv(contri,file = "contri.csv")


contri_plot <- corrplot(var$cos2, is.corr=FALSE)
ggsave("contri_plot.tiff", plot = contri_plot, width =200, height = 200,
       units="mm", dpi=300)

# 1. 
png( 
  filename = "name.png", 
  width = 10,            
  height = 10,           
  units = "in",         
  bg = "white",         
  res = 300)            
# 2. 
corrplot(var$cos2, is.corr=FALSE, number.cex=1)
# 3. 
dev.off()
#no good, but save directly


fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = data$size, # color by groups
             pointsize = 3,
             palette = c("#00AFBB", "#FC4E07"),
             #addEllipses = TRUE, # Concentration ellipses
             legend.title = "Size"
)


fviz_pca_biplot(res.pca, 
                col.ind = data$size, palette = "jco", 
                addEllipses = TRUE, label = "var",
                pointsize = 2.5,
                col.var = "black", repel = TRUE,
                legend.title = "Size") 


fviz_pca_biplot(res.pca, 
                # Individuals
                geom.ind = "point",
                fill.ind = data$size, col.ind = "black",
                pointshape = 21, pointsize = 2,
                palette = "jco",
                addEllipses = TRUE,
                # Variables
                alpha.var ="contrib", col.var = "contrib",
                gradient.cols = "RdYlBu",
                
                legend.title = list(fill = "Species", color = "Contrib",
                                    alpha = "Contrib")
)

T <- fviz_pca_biplot(res.pca, repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)
ggsave("PCA-try.tiff", plot = T, width =130, height = 130,
       units="mm", dpi=300)

P <- fviz_pca_biplot(res.pca, 
               col.ind = data$size, palette = "jco",
               pointsize = 3,
               #addEllipses = TRUE,
               label = "var",
               col.var = "black", repel = TRUE,
               legend.title = "Size") 

ggsave("PCA.tiff", plot = P, width =130, height = 130,
       units="mm", dpi=300)
