library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(dplyr)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/R/scatter_matrix.csv')[,2:9]
colnames(df) <- c("Commercial", "Estates", "High Density", "Leafy Green", "Low Density", "Open Green", "Other Green", "Terraced")

library(psych)
g1 <- pairs.panels(df, 
            method= "pearson",
            hist.col = '#00AFBB',
            density = FALSE,
            #scale=TRUE,
            ellipses = FALSE) 

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/spatial_scatter.tex", width = 4, height = 4)
plot(g1)
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()