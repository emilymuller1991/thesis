library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/sankey_intersection_heatmap_change.csv')[,2:10]
dfm <- melt(df[,colnames(df)],id.vars = 1)
# Give exereme colors:
g1 <- ggplot(dfm, aes(as.factor(variable), clusters_2018_keep_interesting, fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "YlOrRd", direction=1) +
  xlab("2021") +
  ylab("2011") +
  #ggtitle('2011') +
  theme_Publication() +
  coord_fixed() +
  geom_text(aes(label=round(value,2)), size =  1.75) +
  scale_x_discrete(labels= c("Commercial", "Estates", "High-density","Leafy green", "Low-density", "Open green space", "Other green", "Terraced"))
g1
# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  3),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 6),
                 axis.text.x = element_text(size = 6, angle=45, vjust = 0.99, hjust=1),
                 axis.text.y = element_text(size = 6),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0)
)
g1
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/sankey_intersection_heatmap_change.tex", width = 3.2, height = 3.2)
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/sankey_intersection_heatmap_change.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

######################################################################################################################################

df <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/_sankey_intersection_heatmap_change.csv')[,2:10]
dfm <- melt(df[,colnames(df)],id.vars = 1)
# Give exereme colors:
g1 <- ggplot(dfm, aes(as.factor(variable), clusters_2018_keep_interesting, fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "YlOrRd", direction=1) +
  xlab("2021") +
  ylab("2011") +
  #ggtitle('2011') +
  theme_Publication() +
  coord_fixed() +
  geom_text(aes(label=round(value,2)), size =  1.75) +
  scale_x_discrete(labels= c("Commercial", "Estates", "High-density","Leafy green", "Low-density", "Open green space", "Other green", "Terraced"))
g1
# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  3),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 6),
                 axis.text.x = element_text(size = 6, angle=45, vjust = 0.99, hjust=1),
                 axis.text.y = element_text(size = 6),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0)
)
g1
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/_sankey_intersection_heatmap_change.tex", width = 3.2, height = 3.2)
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/_sankey_intersection_heatmap_change.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

