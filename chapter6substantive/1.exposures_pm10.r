library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

exposures <- read.csv('/home/emily/phd/drives/phd/chapter3data/outputs/exposure_pm10a_merge_hierarchicalx.csv')

# dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#008000fd","#ff5555fd","#25e589","#00ff00fd","#d45500fd", "slateblue")
# colours <- c(dend_colours[2], dend_colours[8], dend_colours[1], dend_colours[3], dend_colours[7], dend_colours[4], dend_colours[5], dend_colours[6], "slateblue") 
############################################################################################
dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#008000fd","#ff5555fd","#25e589","#00ff00fd","#d45500fd", "slateblue")
colours <- c(dend_colours[2], dend_colours[1], dend_colours[8], dend_colours[5], dend_colours[6], dend_colours[3], dend_colours[4], dend_colours[7], "slateblue") 
dend_labels <- c("High density", "Commercial", "Low density", "Other green", "Terraced", "Terraced/Low density", "Open greenspace", "Estates", "N/A")
ordered_labels <- c(dend_labels[2], dend_labels[1], dend_labels[8], dend_labels[5], dend_labels[6], dend_labels[3], dend_labels[4], dend_labels[7], "N/A") 
g1 <- ggplot(exposures, aes(x=as.factor(hierarchical8x), y=laei15)) + 
  geom_boxplot(fill=colours, alpha=0.2, outlier.shape = NA) + 
  xlab("Hierarchical Clusters") +
  ylab("Annual PM10") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(limits = c("2", "1", "8", "5", "6", "3", "4", "7", NA),labels=ordered_labels )
g1 <- g1 + theme(axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))
plot(g1)
png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/boxplot_pm10a_hierarhical8_ordered.png", width = 700, height=500)
plot(g1)
dev.off()

############################################################################################
dend_colours <- c(colours[1],colours[2],colours[2],colours[3],colours[4],colours[4],colours[5],colours[6],colours[6],colours[6],colours[7],colours[8],colours[9])
ordered_labels <- c("Commercial", "High density m. Commercial","High-density m. Green","Estates",
                    "Terraced m. Low-density",  "Terraced","Terraced/Low-density",
                    "Low-density m. Other green", "Low density m. Green", "Low-density", 
                    "Other green", "Open greenspace",  "N/A")
g1 <- ggplot(exposures, aes(x=as.factor(hierarchical13x), y=laei15)) + 
  geom_boxplot(fill=dend_colours, alpha=0.2, outlier.shape = NA) + 
  xlab("Hierarchical Clusters") +
  ylab("Annual PM10") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(limits = c("2", "1", "7", "12", "6", "8", "10", "3", "4", "9", "5", "11", NA), labels=ordered_labels )
g1 <- g1 + theme(axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))
plot(g1)
png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/boxplot_pm10a_hierarhical13_ordered.png", width = 700, height=500)
plot(g1)
dev.off()

############################################################################################
dend_colours <- c(colours[1],colours[2],colours[2],colours[2],colours[2],colours[3],colours[3],colours[3],colours[4],
                  colours[4],colours[5],colours[5],colours[5],colours[6],colours[6],colours[6],colours[6],colours[7],
                  colours[7],colours[7],colours[7],colours[8],colours[8],colours[9])
ordered_labels <- c("Commercial","High density m. Commercial", "High-density","High-density m. Terraced","High-density m. Green",
                    "Estates m. High-density", "Estates m. Medium-density", "Estates m. Low-density",
                    "Terraced m. Low-density",  "Terraced", "Terraced/Commercial", "Terraced/HD/Comm", "Terraced/Low-density",
                    "Low-density m. Terraced", "Low-density m. Green", "Low-density", "Low density m. Other green",
                    "Other green m. Green", "Other green", "Other green m. Low-density", "Other green m. Commercial", 
                    "Open greenspace m. Medium-density", "Open greenspace m. Low-density","NA")
g1 <- ggplot(exposures, aes(x=as.factor(hierarchical24x), y=laei15)) + 
  geom_boxplot(fill=dend_colours, alpha=0.2, outlier.shape = NA) + 
  xlab("Hierarchical Clusters") +
  ylab("Annual PM10") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(limits = c("3", "1", "2", "10","8", 
                              "18", "21", "22",
                              "7", "9", "20", "17", "15", 
                              "4", "5", "12", "11",
                              "14", "13", "6", "23",
                              "19", "16", NA), labels=ordered_labels)
g1 <- g1 + theme(axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))
plot(g1)
png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/boxplot_pm10a_hierarhical24_ordered.png", width = 700, height=500)
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