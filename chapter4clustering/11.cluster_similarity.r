library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/matched_images_grouped.csv')
colnames(df) <- c("X",'2011','2018', '2021')
dfm <- melt(df[,c('X','2011','2018', '2021')],id.vars = 1)
colnames(dfm) <- c("X", "Year", "value")
g1 <- ggplot(dfm, aes(x = as.factor(X),y = value)) +
  geom_bar(aes(fill = Year),stat = "identity",position = "dodge") + 
  xlab("Grouped Clusters") +
  ylab("Proportion") +
  #scale_fill_hue(c = 40)  + 
  scale_colour_Publication() + 
  theme_Publication() +
  scale_x_discrete(labels= c('C','E','F','HD','LG','LD','GS','OG','S','T','V'))
#g1 <- g1 + scale_y_continuous(labels = function(x) format(x, scientific = TRUE))
# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/matched_images_grouped.tex", width =4/1.3, height = 4/1.3)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  10),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 8))
                 #axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/matched_images_grouped.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)


df <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/matched_images_ungrouped.csv')
colnames(df) <- c("X",'2011','2018', '2021')
dfm <- melt(df[,c('X','2011','2018', '2021')],id.vars = 1)
colnames(dfm) <- c("X", "Year", "value")
g1 <- ggplot(dfm, aes(x = as.factor(X),y = value)) +
  geom_bar(aes(fill = Year),stat = "identity",position = "dodge") + 
  xlab("Ungrouped Clusters") +
  ylab("Proportion") +
  #scale_fill_hue(c = 40)  + 
  scale_colour_Publication() + 
  theme_Publication()

#g1 <- g1 + scale_y_continuous(labels = function(x) format(x, scientific = TRUE))
# here we open a tex file for output, and set the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/matched_images_ungrouped.tex", width =4/1.3, height = 4/1.3)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  10),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 8))
                 #axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/matched_images_ungrouped.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)
