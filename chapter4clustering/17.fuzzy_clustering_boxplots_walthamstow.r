library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter4clustering/outputs/R/walthamstow_changes_boxplots.csv')[,2:6]
dfm <- melt(df[,c("change", "p_y", "p_x")],id.vars = 1)

# A really basic boxplot.
g1 <- ggplot(dfm, aes(x=as.factor(change), y=value)) + 
  geom_boxplot(fill="slateblue", alpha=0.2) + 
  xlab("Change") +
  ylab("$p$") +
  scale_colour_Publication() + 
  theme_Publication()
g1

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/walthamstow_boxplots.tex", width =3, height = 3)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  7),
                 #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 7))
#axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/walthamstow_boxplots.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

