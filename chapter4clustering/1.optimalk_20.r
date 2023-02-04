library(tidyverse)
library(gridExtra)
library(tikzDevice)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

file <- '/media/emily/south/phd/chapter4clustering/outputs/2018_rmac_cluster_evals.csv'
df <- read.csv(file)

g1 <- ggplot(df, aes(x = as.factor(X), y = distance)) +
  geom_bar(stat = "identity", fill="#999999") +
  xlab("Clusters (C)") +
  ylab("Distance to Centroid") +
  scale_fill_hue(c = 40)  + 
  scale_colour_Publication() + 
  theme_Publication()
g1 

# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/dist_centroids_20.tex", width = 4/1.9, height = 3/1.9)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  5),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5)
                 #,axis.text.x = element_text(angle = 45,vjust = 0.5, hjust=1))
                 )
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()

############################################################################################################ NEW PLOT
g2 <- ggplot(df, aes(x = as.factor(X), y = count)) +
  geom_bar(stat = "identity", fill="#999999") +
  xlab("Clusters (C)") +
  ylab("Points in Cluster") +
  scale_fill_hue(c = 40)  + 
  scale_colour_Publication() + 
  theme_Publication()
  
g2

# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/points_in_cluster_20.tex", width = 4/1.9, height = 3/1.9)

# here we add a LaTeX title to the plot
g2 <- g2 + theme(text = element_text(size =  5),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5)
                 #,axis.text.x = element_text(angle = 45)
                 )
g2
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()

############################################################################################################ NEW PLOT
g3 <- ggplot(df, aes(x = count, y =distance)) +
  geom_point(size = 1, color="#999999") +
  geom_smooth(method=lm) +  
  geom_text(label=as.factor(df$X), size=1.5, vjust = -1, hjust = 1) + 
  xlab("Points in Cluster") +
  ylab("Distance to Centroid") +
  scale_fill_hue(c = 40)  + 
  scale_colour_Publication() + 
  theme_Publication()

g3

# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/scatter_n_dist_20.tex", width = 4/1.9, height = 3/1.9)

# here we add a LaTeX title to the plot
g3 <- g3 + theme(text = element_text(size =  5),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5)
                 #,axis.text.x = element_text(angle = 45)
                 )
g3 <- g3 + scale_x_continuous(labels = function(x) format(x, scientific = TRUE))
g3
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()