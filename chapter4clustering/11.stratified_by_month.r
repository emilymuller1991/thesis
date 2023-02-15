library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df_2011 <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2011_clusters_by_month.csv')
colnames(df_2011) <- c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')
dfm <- melt(df_2011[,c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')],id.vars = 1)
# Give exereme colors:
g1 <- ggplot(dfm, aes(variable, as.factor(month), fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "RdPu")+
  scale_x_discrete(labels= c(0:19)) +
  xlab("Clusters (C)") +
  ylab("Month") +
  #ggtitle('2011') +
  theme_Publication() +
  coord_fixed() +
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 3/1.9))
# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  4),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 4),
                 axis.text.x = element_text(size = 4),
                 axis.text.y = element_text(size = 4),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0)
                )
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2011_clusters_month.tex", width = 4/1.7, height = 3/1.7)
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2011_clusters_month.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

df_2018 <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_clusters_by_month.csv')
colnames(df_2018) <- c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')
dfm <- melt(df_2018[,c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')],id.vars = 1)
# Give exereme colors:
g2 <- ggplot(dfm, aes(variable, as.factor(month), fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "RdPu")+
  scale_x_discrete(labels= c(0:19)) +
  xlab("Clusters (C)") +
  ylab("Month") +
  #ggtitle('2018') +
  theme_Publication() +
  coord_fixed() +
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 3/1.9))
# here we add a LaTeX title to the plot
g2 <- g2 + theme(text = element_text(size =  4),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 4),
                 axis.text.x = element_text(size = 4),
                 axis.text.y = element_text(size = 4),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0)
)
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_clusters_month.tex", width = 4/1.7, height = 3/1.7)
g2
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_clusters_month.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

df_2021 <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2021_clusters_by_month.csv')
colnames(df_2021) <- c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')
dfm <- melt(df_2021[,c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')],id.vars = 1)
# Give exereme colors:
g3 <- ggplot(dfm, aes(variable, as.factor(month), fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "RdPu")+
  scale_x_discrete(labels= c(0:19)) +
  xlab("Clusters (C)") +
  ylab("Month") +
  #ggtitle('2021') +
  theme_Publication() +
  coord_fixed() +
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 3/1.9))
# here we add a LaTeX title to the plot
g3 <- g3 + theme(text = element_text(size =  4),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 4),
                 axis.text.x = element_text(size = 4),
                 axis.text.y = element_text(size = 4),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0),
)
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2021_clusters_month.tex", width = 4/1.7, height = 3/1.7)
g3
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2021_clusters_month.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)