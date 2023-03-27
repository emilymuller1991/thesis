library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

perceptions <- c('beauty', 'boring', 'lively', 'depressing', 'walk', 'safety', 'wealth')
for (perception in perceptions) {
  input <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/2021_",perception,'_perception_by_month.csv')
  df_2011 <- read.csv(input)
  colnames(df_2011) <- c("month",'1','2','3','4','5','6','7','8','9','10')
  dfm <- melt(df_2011[,c("month",'1','2','3','4','5','6','7','8','9','10')],id.vars = 1)
  # Give exereme colors:
  g1 <- ggplot(dfm, aes(variable, as.factor(month), fill= value)) + 
    geom_tile() +
    scale_fill_distiller(palette = "RdPu")+
    scale_x_discrete(labels= c(0:19)) +
    xlab("Deciles") +
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
  path <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/2021_",perception,'_perception_by_month.tex')
  tikzDevice::tikz(file = path, width = 3/1.7, height = 4/1.7)
  plot(g1)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
}

###############################################################################################################################################################
for (perception in perceptions) {
  input <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/2011_",perception,'_perception_by_month.csv')
  df_2011 <- read.csv(input)
  colnames(df_2011) <- c("month",'1','2','3','4','5','6','7','8','9','10')
  dfm <- melt(df_2011[,c("month",'1','2','3','4','5','6','7','8','9','10')],id.vars = 1)
  # Give exereme colors:
  g1 <- ggplot(dfm, aes(variable, as.factor(month), fill= value)) + 
    geom_tile() +
    scale_fill_distiller(palette = "RdPu")+
    scale_x_discrete(labels= c(0:19)) +
    xlab("Deciles") +
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
  path <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/2011_",perception,'_perception_by_month.tex')
  tikzDevice::tikz(file = path, width = 3/1.7, height = 4/1.7)
  plot(g1)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
}

###############################################################################################################################################################
for (perception in perceptions) {
  input <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/both_years_",perception,'_perception_by_month.csv')
  df <- read.csv(input)
  colnames(df) <- c("month", "year",'1','2','3','4','5','6','7','8','9','10')
  dfm <- melt(df[,c("month",'1','2','3','4','5','6','7','8','9','10')],id.vars = 1)
  # Give exereme colors:
  g1 <- ggplot(dfm, aes(variable, as.factor(month), fill= value)) + 
    geom_tile() +
    scale_fill_distiller(palette = "RdPu")+
    scale_x_discrete(labels= c(0:19)) +
    xlab("Deciles") +
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
  path <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/both_years_",perception,'_perception_by_month.tex')
  tikzDevice::tikz(file = path, width = 3/1.7, height = 4/1.7)
  plot(g1)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
  #################################### YEAR PLOTS
  dfm <- melt(df[,c("year",'1','2','3','4','5','6','7','8','9','10')],id.vars = 1)
  # Give exereme colors:
  g1 <- ggplot(dfm, aes(variable, as.factor(year), fill= value)) + 
    geom_tile() +
    scale_fill_distiller(palette = "RdPu")+
    scale_x_discrete(labels= c(0:19)) +
    xlab("Deciles") +
    ylab("Year") +
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
  path <- paste0("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/both_years_",perception,'_perception_by_year.tex')
  tikzDevice::tikz(file = path, width = 3/1.7, height = 4/1.7)
  plot(g1)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
}