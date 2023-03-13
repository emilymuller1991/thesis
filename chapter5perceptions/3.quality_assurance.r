

install.packages("ggplot2")
install.packages("dplyr")
install.packages("broom")
install.packages("ggpubr")
install.packages("nnet")
install.packages("lme4")
install.packages("emmeans")

library(ggplot2)
library(dplyr)
library(broom)
library(ggpubr)
library(nnet)
library(lme4)
library(display)

df <- read.csv("/home/emily/phd/008_web_app/database/data_from_droplet/mlm_data_input_797.csv", header=TRUE)
df$left <- factor(df$left)
df$game <- factor(df$game)
df$london <- factor(df$london)
df$gender <- factor(df$gender)
df$activity <- factor(df$activity)
df$group <- factor (df$group)

m <- glmer(left ~ (1|game), data=df, family=binomial)
summary(m)
lwd = 1.5
a <- lattice::dotplot(ranef(m, condVar=TRUE), 
                    scales = list(tck = c(-1, 0)),
                    par.settings =
                    list(
                        axis.line = list(lwd = lwd),
                        strip.border = list(lwd = lwd), 
                        axis.text=list(cex=2),
                        plot.line = list(lwd = lwd),
                        axis.text.x = element_text(size = 2)
                       ),
                    par.strip.text=list(cex=1), fontsize=2, elinewidth=lwd, cex=1)
tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/mlm_baseline.tex", width = 2.5, height = 2.5)
a
dev.off() 

####################################################################################
df <- read.csv("/home/emily/phd/008_web_app/database/data_from_droplet/mlm_data_input_797.csv", header=TRUE)
df$left <- factor(df$left)
df$game <- factor(df$game)
df$london <- factor(df$london)
df$gender <- factor(df$gender)
df$activity <- factor(df$activity)
df$group <- factor (df$group)

m_act <- glmer(left ~ (activity|game), data=df, family=binomial)
summary(m_act)
lattice::dotplot(ranef(m_act, condVar=TRUE), 
                 scales = list(tck = c(-1, 0), 
                               alternating =  c(1, 1)),
                 par.settings =
                   list(
                     axis.line = list(lwd = lwd),
                     strip.border = list(lwd = lwd), 
                     axis.text=list(cex=2),
                     plot.line = list(lwd = lwd)
                   ),
                 par.strip.text=list(cex=2), fontsize=15, elinewidth=lwd)



m_amt <- glmer(left ~ (group|game), data=df, family=binomial)
summary(m_amt)
lattice::dotplot(ranef(m_amt, condVar=TRUE), 
                 scales = list(tck = c(-1, 0), 
                               alternating =  c(1, 1)),
                 par.settings =
                   list(
                     axis.line = list(lwd = lwd),
                     strip.border = list(lwd = lwd), 
                     axis.text=list(cex=2),
                     plot.line = list(lwd = lwd)
                   ),
                 par.strip.text=list(cex=2), fontsize=15, elinewidth=lwd)

# remove london 2 factor
df <- read.csv("/home/emily/phd/008_web_app/database/data_from_droplet/mlm_london_532.csv", header=TRUE)
df$left <- factor(df$left)
df$game <- factor(df$game)
df$london <- factor(df$london)
df$gender <- factor(df$gender)
df$activity <- factor(df$activity)
df$group <- factor (df$group)

m_lon <- glmer(left ~ (london|game), data=df, family=binomial)
summary(m_lon)
lattice::dotplot(ranef(m_lon, condVar=TRUE), 
                 scales = list(tck = c(-1, 0), 
                               alternating =  c(1, 1)),
                 par.settings =
                   list(
                     axis.line = list(lwd = lwd),
                     strip.border = list(lwd = lwd), 
                     axis.text=list(cex=2),
                     plot.line = list(lwd = lwd)
                   ),
                 par.strip.text=list(cex=2), fontsize=15, elinewidth=lwd)

# remove gender 2 factor
df <- read.csv("/home/emily/phd/008_web_app/database/data_from_droplet/mlm_gender.csv", header=TRUE)
df$left <- factor(df$left)
df$game <- factor(df$game)
df$london <- factor(df$london)
df$gender <- factor(df$gender)
df$activity <- factor(df$activity)
df$group <- factor (df$group)

m_gen <- glmer(left ~ (gender|game), data=df, family=binomial)
summary(m_gen)
lattice::dotplot(ranef(m_gen, condVar=TRUE), 
                 scales = list(tck = c(-1, 0), 
                               alternating =  c(1, 1)),
                 par.settings =
                   list(
                     axis.line = list(lwd = lwd),
                     strip.border = list(lwd = lwd), 
                     axis.text=list(cex=2),
                     plot.line = list(lwd = lwd)
                   ),
                 par.strip.text=list(cex=2), fontsize=15, elinewidth=lwd)








