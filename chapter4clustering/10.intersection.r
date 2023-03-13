library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(networkD3)
library(dplyr)
library(webshot)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

links <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/intersection_sankey.csv')
# From these flows we need to create a node data frame: it lists every entities involved in the flow
nodes <- data.frame(
  name=c(as.character(links$source), as.character(links$target)) %>% 
    unique()
)
# With networkD3, connection must be provided using id, not using real name like in the links dataframe.. So we need to reformat it.
links$IDsource <- match(links$source, nodes$name)-1 
links$IDtarget <- match(links$target, nodes$name)-1

# Add a 'group' column to each connection:
links$group <- as.factor(c(rep("c",6), rep("e",8), rep("hd",7), rep("lg",8),rep("ld",8), rep("gs",5), rep("og",8), rep("t",8)))

# Add a 'group' column to each node. Here I decide to put all of them in the same group to make them grey
nodes$group <- as.factor(c("c","e","hd","lg","ld","gs","og","t","c","hd","lg","ld","og","t","e","gs"))

# Give a color for each group:
# commercial, high-density, leafy green, other green, terraces, estates, open greenspace , low-density
# ff00fffd, 0000fffd, 00fffffd, 008000fd, 0000fffd, d45500fd, 00ff00fd, 00fffffd
my_color <- 'd3.scaleOrdinal() .domain(["c","hd","lg","og","t","e","gs", "ld"]) .range(["#ff00fffd", "#0000fffd", "#008080fd", "#008000fd", "#ff5555fd", "#d45500fd", "#00ff00fd", "#00fffffd"])'

# Make the Network
p <- sankeyNetwork(Links = links, Nodes = nodes, Source = "IDsource", Target = "IDtarget", 
                   Value = "clusters_2018_keep_interesting", NodeID = "name", 
                   colourScale=my_color, LinkGroup="group", NodeGroup="group", fontSize=14)
p <- onRender(
  p,
  '
  function(el,x){
  // select all our node text
  d3.select(el)
  .selectAll(".node text")
  .filter(function(d) { return d.name.endsWith("2018"); })
  .attr("x", x.options.nodeWidth - 22)
  .attr("text-anchor", "end");
  }
  '
)
p <- onRender(
  p,
  '
  function(el,x){
  // select all our node text
  d3.select(el)
  .selectAll(".node text")
  .filter(function(d) { return d.name.endsWith("2021"); })
  .attr("x", x.options.nodeWidth + 10)
  .attr("text-anchor", "start");
  }
  '
)

saveNetwork(p, "sn_.html")
#install phantom:
webshot::install_phantomjs()
# you convert it as png
webshot("sn_.html","/home/emily/phd/drives/phd/chapter4clustering/outputs/R/sankey_intersectio_n.png", vwidth = 600, vheight = 600)