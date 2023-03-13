
library(dendextend)
library(tikzDevice)
# Library
library(tidyverse)
library(dplyr)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df_2011 <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna.csv')[,2:8]
df_2018 <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2018_proportions_all_nonna.csv')[,2:8]
df_2021 <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna.csv')[,2:8]
colnames(df_2011) <-  c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
colnames(df_2018) <-  c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
colnames(df_2021) <-  c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
df <- rbind(df_2011, df_2018, df_2021)


#Clusterisation using 3 variables
df %>%
  dist() %>%
  hclust() %>%
  as.dendrogram() -> dend

clusters <- hclust(dist(df[1:7])) 
plot(clusters)
summary(clusters)
cluster_groups <- cutree(clusters, h=1.1) # 1.1 = 8 # 0.95 = 13
max(cluster_groups)

df_2011_ <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna.csv')
df_2018_ <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2018_proportions_all_nonna.csv')
df_2021_ <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna.csv')
df_2011_$hierarchical8 <- cluster_groups[1:4834]
end <- 4834+4818
df_2018_$hierarchical8 <-  cluster_groups[4835:end]
start <- end + 1
end <- end + 4829
df_2021_$hierarchical8 <-  cluster_groups[start:end]

write.csv(df_2011_, '/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna_hierarchical8_plus2018.csv')
write.csv(df_2018_, '/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna_hierarchical8_plus2018.csv')
write.csv(df_2021_, '/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna_hierarchical8_plus2018.csv')

df$hierarchical8 <- cluster_groups
########################################################################################
#dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#008000fd","#ff5555fd","#25e589","#00ff00fd","#d45500fd")
dend_colours <- c("#a4d47e","#0000fffd","#ff00fffd","#ff5555fd","#d45500fd","#008000fd","#00fffffd","#00ff00fd")
dend_colours <- c(dend_colours[8],dend_colours[7],dend_colours[5],dend_colours[4],dend_colours[3],dend_colours[2],dend_colours[6],dend_colours[1])
dend_ordered <- order.dendrogram(dend)
cluster_groups_dend_ordered <-  cluster_groups[dend_ordered]
# colormap <- numeric(9663)
# for (i in seq_along(cluster_groups)) {
#   dend_i <- dend_ordered[i]
#   cluster <- cluster_groups[dend_i]
#   col_ = dend_colours[cluster]
#   colormap[i] <- col_
# }

dend %>%
  color_branches(clusters = as.numeric(cluster_groups_dend_ordered), col = dend_colours)  %>%
  set("labels_colors", "white") -> dend

plot(dend,axes=TRUE, ylab='Distance')

par(cex=3)
png(filename="/home/emily/phd/drives/phd/chapter4clustering/outputs/R/oa_hierarchical8.png", width = 700, height=700)
plot(dend,axes=TRUE, ylab='Distance', xlab='OA', cex=3)
dev.off()

sub_df <- data.frame()
for (i in seq(8)) {
  sub <- df[df$hierarchical8 == i,][,1:7]
  med <- data.frame(apply(sub, 2, median, na.rm=TRUE))
  sub_ <- t(med)
  sub_ <- cbind(group = i, sub_)
  sub_df <- rbind(sub_df, sub_)
}
sub_df$group <- as.character(sub_df$group)
write.csv(sub_df, '/media/emily/south/phd/chapter4clustering/outputs/R/oa_median_features_for_radar.csv')
########################################################################################
library(ggradar)##
library(tidyverse)
for (c in seq(13)) {
  labels <- c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
  sub <- df[df$hierarchical13 == c,][,1:7]
  med <- data.frame(apply(sub, 2, median, na.rm=TRUE))
  sub_ <- t(med)
  sub_ <- cbind(a = 0, sub_)
  r <- ggradar(
    sub_,
    axis.label.size = 2.5,
    grid.label.size = 2.5,
    group.colours = c(cols[c]),
    values.radar = c("0", "0.6", "0.7"),
    grid.min = 0, grid.mid = 0.6,0.7,
    axis.labels = labels,
    #fill=TRUE,
    gridline.mid.linetype = "solid",
    gridline.min.linetype = "solid",
    background.circle.colour = "white",
    gridline.mid.colour = "grey",
    grid.line.width = 1.5,
    gridline.max.linetype = "solid",
    group.point.size = 2,
  ) 
  #scale_fill_manual(values = c(rep(cols[c], 7)) )
  path <- paste0("/home/emily/phd/drives/phd/chapter4clustering/outputs/R/oa_hierarchical13_radar_",c,'.tex')
  tikzDevice::tikz(file = path, width = 2.5, height = 2.5)
  # r <- r + theme(
  #   text = element_text(size =  5),
  #   # axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
  #   axis.text.y = element_text(size = 5),
  #   axis.text.x = element_text(size=5))
  plot(r)
  dev.off()
  lines <- readLines(con=path)
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=path)
}
