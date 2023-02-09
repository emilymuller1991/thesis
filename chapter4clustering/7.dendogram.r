install.packages("dendextend")
library(dendextend)
# Library
library(tidyverse)

df <- read.csv("/media/emily/south/phd/chapter4clustering/outputs/cluster_centroids.csv")
rownames(df) <- 0:19

# Clusterisation using 3 variables
df %>% 
  dist() %>% 
  hclust() %>% 
  as.dendrogram() -> dend
# Color in function of the cluster
# par(mar=c(1,1,1,7))
dend %>%
  set("labels_col",k=8) %>%
  set("branches_k_color",k = 8)  -> dend
#dend <- dend + xlab("Distance")
# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/dendogram.tex", width = 4, height = 4)

# here we add a LaTeX title to the plot
# dend <- dend + theme(text = element_text(size =  10),
#                  #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
#                  axis.title = element_text(size = 8))
plot(dend,horiz=TRUE, axes=TRUE, xlab='Distance')
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/dendogram.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)
