library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
library(extrafont)
library(ggExtra)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

################################################################################################ SAFETY

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.01True50a68a51fdc9f05596000002_predictions.csv')


g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/safety_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()

################################################################################################ DEPRESSING

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.01True50f62ccfa84ea7c5fdd2e459_predictions.csv')

g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/depressing_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()

################################################################################################ BEAUTY

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.01True5217c351ad93a7d3e7b07a64_predictions.csv')

g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/beauty_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()

################################################################################################ WEALTHY

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.01True50f62cb7a84ea7c5fdd2e458_predictions.csv')

g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/wealthy_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()

################################################################################################ BORING

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.01True50f62c68a84ea7c5fdd2e456_predictions.csv')

g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/boring_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()

################################################################################################ Lively

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.01True50f62c41a84ea7c5fdd2e454_predictions.csv')

g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/lively_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()

################################################################################################

df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/predictions/resnet_epochs_16_lr_0.005Truewalkability_pretrained_beauty_xtra_plots_predictions.csv')

g1 <- ggplot(df, aes(y_pred, y_true), color="#1f77b4",) + geom_point(color="#1f77b4") +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Ground Truth") +
  xlab("Prediction") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/walk_scatter.png", width=700, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
dev.off()
