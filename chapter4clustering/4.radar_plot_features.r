library(dplyr)
library(ggplot2)
library(stringr)
library(readr)
library(tikzDevice)

df_ <- read.csv("/media/emily/south/phd/chapter4clustering/outputs/df_everything_radar.csv")
clusters <- df_[,1:2]$clusters

colors <- c("grey", "green", "red", "black", "white", "grey", "black", "grey", "black", "orange", "green", "red", "yellow", "orange", "red", "black", "red", "blue", "black", "yellow", "black")
colors <- c("white","#999999","green", "red", "#000000", "white","#999999", "#000000","#999999","#000000","#D55E00","green","red","#F0E442","#D55E00","red","#56B4E9", # bicyle
            "red","red","red","#F0E442","red")
            
            
            
  

for(i in seq_along(colnames(df_))) {
  col <- colnames(df_)[i]
  feature <- df_[[col]]
  path <- paste0("/home/emily/phd/drives/phd/chapter4clustering/outputs/R/",col,'.tex')
  tikzDevice::tikz(file = path, width = 1.4, height = 1.4)
  op <- par(cex=0.5)
  plt <- ggplot(select(df_, "clusters", col)) +
    # Make custom panel grid
    geom_hline(
      aes(yintercept = y), 
      data.frame(y = c(0, round(median(feature),3),  round(max(feature),4)+0.0001)),
      color = "lightgrey",
      size = 1
    ) + 
    # Add bars to represent the cumulative track lengths
    # str_wrap(region, 5) wraps the text so each lidf_ne has at most 5 characters
    # (but it doesn't break long words!)
    # Lollipop shaft for mean gain per region
    geom_segment(
      aes(
        x = clusters,
        y = 0,
        xend = clusters,
        yend = round(max(feature),4)+0.0001
      ),
      linetype = "dashed",
      color = "lightgrey"
    ) + 
    geom_col(
      aes(
        x = clusters,
        y = feature,
        fill= rep('i',20)
      ),
      color = 'darkgrey',
  #    size=1.5,
      show.legend = FALSE,
      alpha = 0.6
    ) +
    scale_fill_manual(values=c(rep(colors[i],20))) +
    # Make it circular!
    coord_polar() +
    # Scale y axis so bars don't start in the center
    scale_y_continuous(
      limits = c(-round(max(feature),2)/6, round(max(feature),4)+0.0001),
      expand = c(0, 0),
      breaks = c(0, round(median(feature),3),  round(max(feature),4)+0.0001)
    )+ 
    # Annotate custom scale inside plot
    annotate(
      x = 19.5, 
      y = round(median(feature),3), 
      label = round(median(feature),3), 
      geom = "text", 
      color = "black",
      size=2.5
    ) +
    # annotate(
    #   x = 19.6, 
    #   y = round(max(feature),2), 
    #   label = round(max(feature),2), 
    #   geom = "text", 
    #   color = "black",
    #   size=2.5
    # ) +
    theme(
      # Remove axis ticks and text
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      axis.text.y = element_blank(),
      # Use gray text for the region names
      axis.text.x = element_text(color = "gray12", size = 12),
      # Move the legend to the bottom
      legend.position = "bottom",
    ) +
    theme(    # Make the background white and remove extra grid lines
      panel.background = element_rect(fill = "white", color = "white"),
      panel.grid = element_blank(),
      panel.grid.major.x = element_blank()
      ) +
    scale_x_continuous(labels = as.character(clusters), breaks = 0:19)
  # here we add a LaTeX title to the plot
  plt <- plt + theme(
    #text = element_text(size =  4),
                  # axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                   #axis.title = element_text(size = 5)
                   axis.text.x = element_text(size=5))
  
  plot(plt)
  # closing the graphics device saves the file we opened with tikzDevice::tikz
  dev.off()
  # remove all lines that invisibly mess up the bounding box
  lines <- readLines(con=path)
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=path)
}


#library(tidyverse)
#library(scales)
#library(ggplot2)
#library(grid)
#install.packages("devtools")
#library(devtools)
#devtools::install_github("ricardo-bion/ggradar", 
#                         dependencies = TRUE)
#library(ggradar)##

#ggradar(
#  df_[1, ], 
#  values.radar = c("0", round(as.vector(apply(df_[1,2:21],1,median)),3),  round(max(df_[1,2:21]),2)),
#  grid.min = 0, grid.mid = as.vector(apply(df_[1,2:21],1,median)), grid.max = max(df_[1,2:21]),
#  axis.labels = categories,
#  fill=TRUE,
#  gridline.mid.linetype = "solid",
#  gridline.min.linetype = "solid",
#  background.circle.colour = "white",
#  gridline.mid.colour = "black",
#  grid.line.width = 1.5,
#  gridline.max.linetype = "solid",
#  group.point.size = 4,
#  group.colours = c("#999999", "#E69F00", "#56B4E9", "green", "#F0E442", "#0072B2", "red", "#CC79A7")
#)##