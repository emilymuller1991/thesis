library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
library(extrafont)
library(ggExtra)
library(ggpubr)
library(stargazer)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/all_scores_all_residuals_le.csv')
m1 <- lm(df$Life.expectancy..Male..central.estimate. ~ df$income + df$wealth_resid + df$education + df$education_resid + df$outdoor + df$outdoor_resid )  #Create a linear model
m2 <- lm(df$Life.expectancy..Female..central.estimate. ~ df$income + df$wealth_resid + df$education + df$education_resid + df$outdoor + df$outdoor_resid )  #Create a linear model
stargazer(m1, m2, align=TRUE)

summary(m1)

df_ <- df[c('income', 'Life.expectancy..Female..central.estimate.')]
dfm <- melt(df_[,colnames(df_)],id.vars = 1)
g1 <- ggplot(dfm, aes(income, value)) + 
  geom_point() +
  geom_smooth(method='lm')
plot(g1)

df_ <- df[c('wealth_resid', 'Life.expectancy..Female..central.estimate.')]
dfm <- melt(df_[,colnames(df_)],id.vars = 1)
g1 <- ggplot(dfm, aes(wealth_resid, value)) + 
  geom_point() +
  geom_smooth(method='lm')
plot(g1)

df_ <- df[c('wealth_resid', 'income')]
dfm <- melt(df_[,colnames(df_)],id.vars = 1)
g1 <- ggplot(dfm, aes(wealth_resid, value)) + 
  geom_point() +
  geom_smooth(method='lm')
plot(g1)