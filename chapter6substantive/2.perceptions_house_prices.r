library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
library(extrafont)
library(ggExtra)
library(ggpubr)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/house_prices_wealth_perception.csv')
m1 <- lm(df$Unnamed..0~df$Price..central.estimate..natural.log.scale.)  #Create a linear model
df$resid <- resid(m1)

g1 <- ggplot(df, aes(Price..central.estimate..natural.log.scale., Unnamed..0)) + 
  geom_point(aes(color = resid), size=2) + 
  scale_color_gradient2(low = "red", mid = "grey", high = "yellow", midpoint = 0) +
  geom_smooth(method='lm') +
  #scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Wealth Perception OA 2021") +
  xlab("Price (central estimate, natural log scale)") 
#g1 <- g1 + stat_cor(method="pearson") 
plot(g1)

png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/scatterplot_house_prices_wealth_perceptions_res.png", width = 700, height=700)
plot(g1)
dev.off()
write.csv(df, '/media/emily/south/phd/chapter6substantive/outputs/wealth_perceptions_oa_residuals.csv')

df <- read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/house_prices_wealth_perception_lsoa.csv')
m1 <- lm(df$Unnamed..0~df$Price..central.estimate..natural.log.scale.)  #Create a linear model
df$resid <- resid(m1)

g1 <- ggplot(df, aes(Price..central.estimate..natural.log.scale., Unnamed..0)) + 
  geom_point(aes(color = resid), size=2) + 
  scale_color_gradient2(low = "red", mid = "grey", high = "yellow", midpoint = 0) +
  geom_smooth(method='lm') +
  #scale_colour_Publication() + 
  theme_Publication()  +
  ylab("Wealth Perception LSOA 2021") +
  xlab("Price (central estimate, natural log scale)") 
#g1 <- g1 + stat_cor(method="pearson") 
plot(g1)

png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/scatterplot_house_prices_wealth_perceptions_lsoa_res.png", width = 700, height=700)
plot(g1)
dev.off()
write.csv(df, '/media/emily/south/phd/chapter6substantive/outputs/wealth_perceptions_lsoa_residuals.csv')

# g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
#                  axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
#                  axis.title = element_text(size = 40),
#                  axis.line = element_line(colour = 'black', size = 2))
# 
# png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/safety_scatter.png", width=700, height=700)
# par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
# ggExtra::ggMarginal(g1, type = "histogram", fill="#1f77b4")
# dev.off()
