library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
library(extrafont)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')


df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/walk_ranks.csv')
df$order <- 1:24522
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=X0-X1, ymax=X0+X1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,X0), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))
  
g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/walk_ranks.png", width=600, height=700)
plot(g1)
dev.off()


####################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/More_beautiful_ranks.csv')
df$order <- 1:111390
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=trueskill.score-trueskill.stds..1, ymax=trueskill.score+trueskill.stds..1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,trueskill.score), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/beauty_ranks.png", width=600, height=700)
plot(g1)
dev.off()

####################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/More_depressing_ranks.csv')
df$order <- 1:111390
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=trueskill.score-trueskill.stds..1, ymax=trueskill.score+trueskill.stds..1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,trueskill.score), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")
g1
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/depressing_ranks.png", width=600, height=700)
plot(g1)
dev.off()
####################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/Wealthier_ranks.csv')
df$order <- 1:111390
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=trueskill.score-trueskill.stds..1, ymax=trueskill.score+trueskill.stds..1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,trueskill.score), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")
g1
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/wealthy_ranks.png", width=600, height=700)
plot(g1)
dev.off()

####################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/Boring_ranks.csv')
df$order <- 1:111390
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=trueskill.score-trueskill.stds..1, ymax=trueskill.score+trueskill.stds..1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,trueskill.score), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")
g1
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/boring_ranks.png", width=600, height=700)
plot(g1)
dev.off()

####################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/Safer_ranks.csv')
df$order <- 1:111390
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=trueskill.score-trueskill.stds..1, ymax=trueskill.score+trueskill.stds..1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,trueskill.score), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")
g1
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/safety_ranks.png", width=600, height=700)
plot(g1)
dev.off()

####################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/Livelier_ranks.csv')
df$order <- 1:111390
g1 <- ggplot(df) + 
  geom_errorbar(aes(x=order,ymin=trueskill.score-trueskill.stds..1, ymax=trueskill.score+trueskill.stds..1), width=.2,
                position=position_dodge(0.05), color="#1f77b4", alpha=0.01) +
  geom_point(aes(order,trueskill.score), color="#1f77b4") +
  coord_flip()  +
  scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Q-score") +
  xlab("")
g1
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2))

g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/lively_ranks.png", width=600, height=700)
plot(g1)
dev.off()
