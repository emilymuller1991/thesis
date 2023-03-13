library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/walthamstow_heatmap.csv')[,2:7]
#colnames(df_2011) <- c("month",'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19')
dfm <- melt(df[,colnames(df)],id.vars = 1)
# Give exereme colors:
g1 <- ggplot(dfm, aes(as.factor(variable), clusters_2011_cleanup, fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "RdPu")+
  scale_x_discrete(labels= c(0:19)) +
  xlab("2021") +
  ylab("2011") +
  #ggtitle('2011') +
  theme_Publication() +
  coord_fixed() +
  geom_text(aes(label=round(value,1))) +
  #guides(fill = guide_colourbar(barwidth = 0.5, barheight = 3/1.9))  +
  scale_x_discrete(labels= c("Estates", "High-density", "Low-density", "Other green", "Terraced"))
g1
# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  6),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 7),
                 axis.text.x = element_text(size = 7, angle=45, vjust = 0.99, hjust=1),
                 axis.text.y = element_text(size = 7),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0)
                )
g1
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/walthamstow_heatmap.tex", width = 4, height = 4)
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/walthamstow_heatmap.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)