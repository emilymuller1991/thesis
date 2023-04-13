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

df <- read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/2019_deprivation_scores_london_london_perception_wealth.csv')
m1 <- lm(df$Unnamed..0~df$Income.Score..rate.)  #Create a linear model
df$resid <- resid(m1)

g1 <- ggplot(df, aes(Income.Score..rate., Unnamed..0)) + 
  geom_point(aes(color = resid)) + 
  #guides(fill = guide_colourbar(title.position = "top")) +
  scale_color_gradient2(low = "red", mid = "grey", high = "yellow", midpoint = 0, name='Resid') +
  geom_smooth(method='lm') +
  #scale_colour_Publication() + 
  theme_Publication()  +
  stat_cor(size=10) +
  ylab("Wealth Perception Score") +
  xlab("Income Deprivation Score")
#g1 <- g1 + stat_cor(method="pearson") 
plot(g1)


# here# we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2),
                 legend.position="right",
                 legend.direction = "vertical",
                 legend.title.align = 0.5,
                 legend.key.height=unit(1.6, "cm")
      ) 

plot(g1)

png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/scatterplot_income_deprivation_wealth_perceptions_res.png", width = 800, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
plot(g1)
dev.off()

# write.csv(df, '/media/emily/south/phd/chapter6substantive/outputs/wealth_perception_income_deprivation_residuals.csv')

################################################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/deprivation_merge_perceptions.csv')
m1 <- lm(df$depressing~df$Education..Skills.and.Training.Score)  #Create a linear model
df$resid <- resid(m1)

g1 <- ggplot(df, aes(Education..Skills.and.Training.Score, depressing)) + 
  geom_point(aes(color = resid)) + 
  scale_color_gradient2(low = "yellow", mid = "grey", high = "red", midpoint = 0, name="Resid") +
  geom_smooth(method='lm') +
  #scale_colour_Publication() + 
  theme_Publication()  +
  stat_cor(size=10) +
  ylab("Depressing Perception Score") +
  xlab("Education Deprivation Score") 
#g1 <- g1 + stat_cor(method="pearson") 
plot(g1)
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2),
                 legend.position="right",
                 legend.direction = "vertical",
                 legend.title.align = 0.5,
                 legend.key.height=unit(1.6, "cm")
) 

plot(g1)


png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/scatterplot_education_deprivation_depressing_perceptions_res.png", width = 800, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
plot(g1)
dev.off()
write.csv(df, '/media/emily/south/phd/chapter6substantive/outputs/depressing_perception_education_deprivation_residuals.csv')

################################################################################################################################
df <- read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/2019_deprivation_scores_london_london_perception_boring.csv')
m1 <- lm(df$Unnamed..0~df$Outdoors.Sub.domain.Score)  #Create a linear model
df$resid <- resid(m1)

g1 <- ggplot(df, aes(Outdoors.Sub.domain.Score, Unnamed..0)) + 
  geom_point(aes(color = resid)) + 
  scale_color_gradient2(low = "yellow", mid = "grey", high = "red", midpoint = 0, name='Resid') +
  geom_smooth(method='lm') +
  #scale_colour_Publication() + 
  theme_Publication()  +
  stat_cor(size=10) +
  ylab("Boring Perception Score") +
  xlab("Outdoor Deprivation Score") 

#g1 <- g1 + stat_cor(method="pearson") 
g1 <- g1 + theme(text = element_text(size =  40, family="LM Roman 10"),
                 axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 0, l = 10)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2),
                 legend.position="right",
                 legend.direction = "vertical",
                 legend.title.align = 0.5,
                 legend.key.height=unit(1.6, "cm")
) 

plot(g1)

png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/scatterplot_outdoor_deprivation_boring_perceptions_res.png", width = 800, height=700)
par(omi = c(0,0,0,0), mgp = c(0,0,0), mar = c(0,0,0,0))
plot(g1)
dev.off()
write.csv(df, '/media/emily/south/phd/chapter6substantive/outputs/boring_perception_living_env_dep_residuals.csv')



