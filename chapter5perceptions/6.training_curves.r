library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
library(extrafont)
library(ggExtra)
library(colorspace)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/training_curves_wandb.csv')[1:15]
colnames(df) <- c('X', 'Walk Train', 'Walk Val', 'Boring Train', 'Boring Val', 'Lively Train', 'Lively Val', 'Wealth Train', 'Wealth Val', 
                  'Depressing Train', 'Depressing Val','Beauty Train', 'Beauty Val', 'Safety Train', 'Safety Val')
dfm <- melt(df[,c('X', 'Walk Train', 'Walk Val', 'Boring Train', 'Boring Val', 'Lively Train', 'Lively Val', 'Wealth Train', 'Wealth Val', 
                  'Depressing Train', 'Depressing Val','Beauty Train', 'Beauty Val', 'Safety Train', 'Safety Val')],id.vars = 1)
dfm$val_train <- rep(c(rep('Train', 16), rep('Val', 16)),7)

colors <- sort(rep(rainbow_hcl(7),2))
g1 <- ggplot(dfm, aes(x = X, y = value)) + 
  geom_line(aes(color = variable, linetype=as.factor(val_train))) +
  xlab("Epochs") +
  ylab("MSE") +
  scale_colour_Publication() + 
  theme_Publication() +
  # scale_color_manual(values = c(colors),breaks = c("Walk Train", "Boring Train","Lively Train", "Wealth Train", "Depressing Train","Beauty Train", "Safety Train"))
  scale_color_manual(name = 'Perception',
                     values = c('Walk Train' = colors[1],
                                'Walk Val' = colors[1], 
                                'Boring Train'= colors[3], 
                                'Boring Val'= colors[3],
                                'Lively Train'=colors[5], 
                                'Lively Val'== colors[6],
                                'Wealth Train'== colors[7],
                                'Wealth Val'= colors[8], 
                                'Depressing Train'= colors[9],
                                'Depressing Val'= colors[10],
                                'Beauty Train'= colors[11],
                                'Beauty Val'= colors[12],
                                'Safety Train'= colors[13],
                                'Safety Val'= colors[14]),
                     breaks = c("Walk Train", "Boring Train","Lively Train", "Wealth Train", "Depressing Train","Beauty Train", "Safety Train"),
                     labels = c("Walk", "Boring","Lively", "Wealth", "Depressing","Beauty", "Safety")) +
  scale_linetype_manual(" ",values=c("Train"=2,"Val"=1))
g1

# here we open a tex file for output, and set hcl_palettes(palette = "set2")the plots dimensions
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/training_curves.tex", width = 4, height = 4)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  10),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 8),
                 legend.text = element_text(size= 6 ) ,
                 legend.position = "right",
                 legend.direction = 'vertical'
                 )
plot(g1)
dev.off()