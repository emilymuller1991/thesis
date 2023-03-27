library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

perceptions <- c('beauty', 'boring', 'lively', 'depressing', 'walk', 'safety', 'wealth')
for (perception in perceptions) {
  input <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/lsa_change_in_deciles_",perception,'_perception.csv')
  df <- read.csv(input)[,2:12]
  colnames(df) <- c("decile_2011",'1','2','3','4','5','6','7','8','9','10')
  df$decile_2011 <- c(1,2,3,4,5,6,7,8,9,10)
  dfm <- melt(df[,colnames(df)],id.vars = 1)
  # Give exereme colors:
  g1 <- ggplot(dfm, aes(as.factor(variable), as.factor(decile_2011), fill= value)) + 
    geom_tile() +
    scale_fill_distiller(palette = "YlOrRd", direction=1) +
    xlab("2021") +
    ylab("2011") +
    #ggtitle('2011') +
    theme_Publication() +
    coord_fixed() +
    geom_text(aes(label=round(value,2)), size =  1.75) #+
    #scale_y_discrete(labels= c('1','2','3','4','5','6','7','8','9','10'))
  g1
  # here we add a LaTeX title to the plot
  g1 <- g1 + theme(text = element_text(size =  3),
                  axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                  axis.title = element_text(size = 6),
                  axis.text.x = element_text(size = 6),
                  axis.text.y = element_text(size = 6),
                  axis.line = element_line(size=0),
                  legend.position = "none",
                  legend.direction = "vertical",
                  legend.title	= element_text(size=0)
  )
  g1
  path <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/lsoa_change_in_decile_",perception,'_perception.tex')
  tikzDevice::tikz(file = path, width = 3.2, height = 3.2)
  plot(g1)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
}

######################################################################################################################
perceptions <- c('beauty', 'boring', 'lively', 'depressing', 'walk', 'safety', 'wealth')
for (perception in perceptions) {
  input <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/lsoa_change_in_deciles_BOTH_",perception,'_perception.csv')
  df <- read.csv(input)[,2:12]
  colnames(df) <- c("decile_2011",'1','2','3','4','5','6','7','8','9','10')
  df$decile_2011 <- c(1,2,3,4,5,6,7,8,9,10)
  dfm <- melt(df[,colnames(df)],id.vars = 1)
  # Give exereme colors:
  g1 <- ggplot(dfm, aes(as.factor(variable), as.factor(decile_2011), fill= value)) + 
    geom_tile() +
    scale_fill_distiller(palette = "YlOrRd", direction=1) +
    xlab("2021") +
    ylab("2011") +
    #ggtitle('2011') +
    theme_Publication() +
    coord_fixed() +
    geom_text(aes(label=round(value,2)), size =  1.75) #+
  #scale_y_discrete(labels= c('1','2','3','4','5','6','7','8','9','10'))
  g1
  # here we add a LaTeX title to the plot
  g1 <- g1 + theme(text = element_text(size =  3),
                   axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 0)), 
                   axis.title = element_text(size = 6),
                   axis.text.x = element_text(size = 6),
                   axis.text.y = element_text(size = 6),
                   axis.line = element_line(size=0),
                   legend.position = "none",
                   legend.direction = "vertical",
                   legend.title	= element_text(size=0)
  )
  g1
  path <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/lsoa_change_in_decile_BOTH_",perception,'_perception.tex')
  tikzDevice::tikz(file = path, width = 3.2, height = 3.2)
  plot(g1)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
}
