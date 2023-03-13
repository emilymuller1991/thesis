library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df_2018 <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_distances_to_centroid.csv')
radii <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_radii.csv')

upper_whisker <- function(x) {
  q <- quantile(x, probs = c(0.25, 0.75), na.rm= TRUE)
  iqr <- q[2] - q[1]
  upper <- q[2] + 1.5*iqr
  upper_adj <- min(max(x[x <= upper], na.rm = TRUE), upper)
  return(upper_adj)
}

lower_whisker <- function(x) {
  q <- quantile(x, probs = c(0.25, 0.75), na.rm= TRUE)
  iqr <- q[2] - q[1]
  lower <- q[1] - 1.5*iqr
  lower_adj <- max(min(x[x >= lower], na.rm = TRUE), lower)
  return(lower_adj)
}

# A really basic boxplot.
g1 <- ggplot(df_2018, aes(x=as.factor(clusters), y=point_centroid)) + 
  geom_boxplot(fill="slateblue", alpha=0.2, outlier.shape = NA) + 
  xlab("Clusters") +
  ylab("2018 Distance to Centroid") +
  scale_colour_Publication() + 
  theme_Publication() +
  stat_summary(fun=upper_whisker,  geom = "point",  size=1, color="red") +
  stat_summary(fun=lower_whisker,  geom = "point",  size=1, color="red")
  
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_distances_to_centroid.tex", width =4/1.9, height = 4/1.9)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  5),
                 #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5))
#axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_distances_to_centroid.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

##############################################################################

df_years <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/both_years_distances_to_centroid.csv')

g1 <- ggplot(df_years, aes(x=as.factor(clusters), y=distance)) + 
  geom_boxplot(fill="slateblue", alpha=0.2, outlier.shape = NA) + 
  xlab("Clusters") +
  ylab("2011 + 2021 Distance to Centroid") +
  scale_colour_Publication() + 
  theme_Publication() 

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/both_years_distances_to_centroid.tex", width =4/1.9, height = 4/1.9)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  5),
                 #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5))
#axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/both_years_distances_to_centroid.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

##############################################################################

g1 <- ggplot(df_years, aes(x=as.factor(clusters), y=p)) + 
  geom_boxplot(fill="skyblue", alpha=0.2, outlier.shape = NA) + 
  xlab("Clusters") +
  ylab("2011 + 2021 Fuzzy Clustering (p)") +
  scale_colour_Publication() + 
  theme_Publication() 
g1

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/both_years_p.tex", width =4/1.9, height = 4/1.9)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  5),
                 #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5))
#axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/both_years_p.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)
