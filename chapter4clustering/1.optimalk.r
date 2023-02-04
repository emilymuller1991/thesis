library(tidyverse)
library(gridExtra)
library(tikzDevice)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')
# install.packages("tidyverse")
# library(dplyr)
# library(ggplot2)
# library(tidyverse)

prefix <- "/run/user/1000/gvfs/smb-share:server=rds.imperial.ac.uk,share=rds/user/emuller/home/emily/phd/003_image_matching/clustering/output/2018/"

dataframes <- list()

for (k in c(4,6,8,9,10,12,14,16,18,20,25,30)) {
  df <- read.csv(paste0(prefix, "metrics_for_", k, "_clusters.csv"))
  dataframes[[length(dataframes) + 1]] <- df
}

df <- bind_rows(dataframes)

g1 <- ggplot(df, aes(x = k, y = dist)) +
  geom_line(size=1.5) +
  geom_point(size=1.5) + 
  xlab("Number of Clusters (K)") +
  ylab("Total Distance to Centroid") +
  scale_colour_Publication() + 
  theme_Publication()

g1 <- g1 + scale_y_continuous(labels = function(x) format(x, scientific = TRUE))
g1 

# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/dist_centroids.tex", width = 4, height = 3)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  10),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 8))
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

g2 <- ggplot(df, aes(x = k, y = scs)) +
  geom_line(color="#CC79A7", size=1.5) +
  geom_point(color="#CC79A7", size=1.5) + 
  xlab("Number of Clusters (K)") +
  ylab("Silhouette Score") +
  scale_colour_Publication() + 
  theme_Publication()
g2
# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/sil_score_resize1.5.tex", width =4/1.3, height = 3/1.3)

# here we add a LaTeX title to the plot
g2 <- g2 + theme(text = element_text(size =  10),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 8))
g2
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()


g3 <- ggplot(df, aes(x = k, y = dbs)) +
  geom_line(color="#56B4E9", size=1.5) +
  geom_point(color="#56B4E9", size=1.5) + 
  xlab("Number of Clusters (K)") +
  ylab("Davies-Bouldin Score") +
  scale_colour_Publication() + 
  theme_Publication()

g3
# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/db_score_resize1.5.tex", width = 4/1.3, height = 3/1.3)

# here we add a LaTeX title to the plot
g3 <- g3 + theme(text = element_text(size =  10),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 8))
g3
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
