library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_walk_ranks.csv')
df$order <- 1:613

g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=X0-X1, ymax=X0+X1), width=.2,
                position=position_dodge(0.05), color="slateblue", alpha=0.2) +
  geom_point(aes(order,X0), color="slateblue") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  3),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                 axis.title = element_text(size = 5),
                 axis.text.x = element_text(size = 5),
                 axis.text.y = element_text(size = 5),
                 panel.border = element_blank(),
                 legend.position = "none")

g1
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_walk_rank.tex", width = 1.75, height = 2)
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/pilot_walk_rank.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)