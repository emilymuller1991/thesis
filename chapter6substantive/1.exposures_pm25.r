library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

exposures <- read.csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_pm25_merge_hierarchicalx.csv')

dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#008000fd","#ff5555fd","#25e589","#00ff00fd","#d45500fd", "slateblue")
colours <- c(dend_colours[2], dend_colours[8], dend_colours[1], dend_colours[3], dend_colours[7], dend_colours[4], dend_colours[5], dend_colours[6], "slateblue") 
############################################################################################
g1 <- ggplot(exposures, aes(x=as.factor(hierarchical8x), y=laei15)) + 
  geom_boxplot(fill=dend_colours, alpha=0.2, outlier.shape = NA) + 
  xlab("Hierarchical Clusters") +
  ylab("PM2.5 exceedance") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(labels=c("High density", "Commercial", "Low density", "Other green", "Terraced", "Terraced/Low density", "Open greenspace", "Estates", "N/A"))
g1 <- g1 + theme(axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))
plot(g1)
#png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/boxplot_pm25e_hierarhical8.png", width = 700, height=500)
plot(g1)
dev.off()

############################################################################################
dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#00fffffd","#008000fd","#ff5555fd","#0000fffd","#ff5555fd","#00fffffd","#25e589","#00ff00fd","#d45500fd","#ff00fffd")
#level_order <- c("1","7","2","13","3","4","9","5","6","8","10","11","12")
level_order <- c(rep(1,168),rep(7,262),rep(2,28),rep(3,1122),rep(4,278),rep(9,471),rep(5,194),rep(6,539),rep(8,97),rep(10,203),rep(11,98),rep(12,64))
exposures$hierarchical13x <- factor(exposures$hierarchical13x, levels=level_order)

g1 <- ggplot(exposures, aes(x=as.factor(hierarchical13x), y=laei15)) + 
  geom_boxplot(fill=dend_colours, alpha=0.2, outlier.shape = NA) + 
  xlab("Hierarchical Clusters") +
  ylab("PM2.5 exceedance") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(labels=c("High density m. Commercial", "Commercial", "Low-density m. Other green", "Low density m. Green", "Other green", "Terraced m. Low-density", "High-density m. Green", "Terraced",
                            "Low-density", "Terraced/Low-density", "Open greenspace", "Estates","Commercial E&C", "N/A"))
g1 <- g1 + theme(axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))
plot(g1)
png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/boxplot_pm25e_hierarhical13.png", width = 700, height=500)
plot(g1)
dev.off()

############################################################################################
dend_colours <- c("#0000fffd","#0000fffd","#ff00fffd","#00fffffd","#00fffffd","#008000fd","#ff5555fd","#0000fffd","#ff5555fd","#0000fffd","#00fffffd","#00fffffd","#008000fd","#008000fd","#25e589","#00ff00fd","#25e589","#d45500fd","#00ff00fd","#25e589","#d45500fd","#d45500fd","#008000fd","#ff00fffd")
g1 <- ggplot(exposures, aes(x=as.factor(hierarchical24x), y=laei15)) + 
  geom_boxplot(fill=dend_colours, alpha=0.2, outlier.shape = NA) + 
  xlab("Hierarchical Clusters") +
  ylab("PM2.5 exceedance") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(labels=c("High density m. Commercial", "High-density", "Commercial","Low-density m. Terraced", "Low-density m. Green", "Other green m. Green", 
                            "Terraced m. Low-density", "High-density m. Green", "Terraced", "High-density m. Terraced", "Low density m. Other green","Low-density", "Other green m. Low-density", "Other green",
                            "Terraced/Low-density","Open greenspace m. Low-density","Terraced/HD/Comm","Estates m. Low-density","Open greenspace m. Medium-density",
                            "Terraced/Commercial", "Estates m. High-density", "Estates m. Medium-density", "Other green m. Commcerial", "Commercial E&C"))
g1 <- g1 + theme(axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))
plot(g1)
png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/boxplot_pm10a_hierarhical24.png", width = 700, height=500)
plot(g1)
dev.off()









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