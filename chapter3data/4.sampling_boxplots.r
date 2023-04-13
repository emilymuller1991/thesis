library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df_oa <- read.csv('/home/emily/phd/drives/phd/chapter3data/outputs/sampling_rate_oa_all_years_sql.csv')[,1:4]
dfm <- melt(df_oa[,colnames(df_oa)],id.vars = 1)

# A really basic boxplot.
g1 <- ggplot(dfm, aes(x=as.factor(variable), y=value)) + 
  geom_boxplot(fill="slateblue", alpha=0.2, outlier.shape = NA) + 
  xlab("Year Dataset") +
  ylab("Sampling Rate OA") +
  scale_colour_Publication() + 
  theme_Publication() +
  scale_y_continuous(limits = c(0, 2))
g1

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter3data/outputs/R/yearly_oa_sampling_rate.tex", width =4/1.9, height = 4/1.9)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  5),
                 #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 5))
#axis.text.x = element_text(angle = 45,vjust = 0.99, hjust=1))

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/2018_distances_to_centroid.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

##############################################################################