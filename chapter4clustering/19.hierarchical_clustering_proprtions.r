
library(dendextend)
library(tikzDevice)
# Library
library(tidyverse)
library(dplyr)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

df_2011 <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna.csv')[,2:8]
df_2021 <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna.csv')[,2:8]
colnames(df_2011) <-  c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
colnames(df_2021) <-  c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
df <- rbind(df_2011, df_2021)


#Clusterisation using 3 variables
df %>%
  dist() %>%
  hclust() %>%
  as.dendrogram() -> dend

#dend <- dend + xlab("Distance")
# # here we open a tex file for output, and set the plots dimensions
# tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/oa_dendogram.tex", width = 4, height = 4)
# 
# # here we add a LaTeX title to the plot
# # dend <- dend + theme(text = element_text(size =  10),
# #                  #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),
# #                  axis.title = element_text(size = 8))
plot(dend,horiz=TRUE, axes=TRUE, xlab='Distance')
# # closing the graphics device saves the file we opened with tikzDevice::tikz
# dev.off()
# path <- "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/oa_dendogram.tex"
# lines <- readLines(con=path)
# lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
# lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
# writeLines(lines,con=path)

clusters <- hclust(dist(df[1:7])) 
plot(clusters)
summary(clusters)
cluster_groups <- cutree(clusters, h=0.655) # 1 = 8 # 0.785 = 13 # 0.655 = 24
max(cluster_groups)

df_2011_ <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna.csv')
df_2021_ <- read.csv('/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna.csv')
df_2011_$hierarchical8 <- cluster_groups[1:4834]
end <- 4834+4829
df_2021_$hierarchical8 <-  cluster_groups[4835:end]
# 
write.csv(df_2011_, '/media/emily/south/phd/chapter4clustering/outputs/2011_proportions_all_nonna_lsoa_hierarchical24.csv')
write.csv(df_2021_, '/media/emily/south/phd/chapter4clustering/outputs/2021_proportions_all_nonna_lsoa_hierarchical24.csv')

df$hierarchical8 <- cluster_groups

dend %>%
  color_branches(k = 13)  %>%
  set("labels_colors", "white") -> dend
plot(dend)
########################################################################################
dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#008000fd","#ff5555fd","#25e589","#00ff00fd","#d45500fd")
dend_colours <- c(dend_colours[5],dend_colours[2],dend_colours[1],dend_colours[3],dend_colours[7],dend_colours[4],dend_colours[8],dend_colours[6])
dend_ordered <- order.dendrogram(dend)
cluster_groups_dend_ordered <-  cluster_groups[dend_ordered]
colormap <- numeric(9663)
# for (i in seq_along(cluster_groups)) {
#   dend_i <- dend_ordered[i]
#   cluster <- cluster_groups[dend_i]
#   col_ = dend_colours[cluster]
#   colormap[i] <- col_
# }

dend %>%
  color_branches(k = 13)  %>%
  set("labels_colors", "white") -> dend
plot(dend)

dend %>%
  color_branches(clusters = as.numeric(cluster_groups_dend_ordered), col = dend_colours)  %>%
  set("labels_colors", "white") -> dend

# par(cex=3)
# png(filename="/home/emily/phd/drives/phd/chapter4clustering/outputs/R/lsoa_hierarchical8.png", width = 700, height=700)
plot(dend,axes=TRUE, ylab='Distance')
# dev.off()

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter4clustering/outputs/R/oa_dendogram.tex", width = 4, height = 4)
# 
# here we add a LaTeX title to the plot
dend <- dend + theme(text = element_text(size =  10),
                 #axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),
                 axis.title = element_text(size = 8))
plot(dend, axes=TRUE, xlab='Distance')
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()

c=1
sub <- df[df$hierarchical8 == c,][,1:7]
med <- data.frame(apply(sub, 2, median, na.rm=TRUE))
sub_ <- t(med)
sub_ <- cbind(a = 0, sub_)

sub_df <- data.frame()
for (i in seq(max(cluster_groups))) {
  sub <- df[df$hierarchical8 == i,][,1:7]
  med <- data.frame(apply(sub, 2, median, na.rm=TRUE))
  sub_ <- t(med)
  sub_ <- cbind(group = i, sub_)
  sub_df <- rbind(sub_df, sub_)
}
sub_df$group <- as.character(sub_df$group)
write.csv(sub_df, '/media/emily/south/phd/chapter4clustering/outputs/R/lsoa_median_features_for_radar_hierarchical24.csv')
# write.csv(sub_df, '/media/emily/south/phd/chapter4clustering/outputs/R/lsoa_median_features_for_radar.csv')

library(ggradar)
library(dplyr)
library(scales)
library(tibble)

mtcars_radar <- mtcars %>% 
  as_tibble(rownames = "group") %>% 
  mutate_at(vars(-group), rescale) %>% 
  tail(4) %>% 
  select(1:4)
mtcars_radar$group <- as.factor(mtcars_radar$group )
ggradar(mtcars_radar)



########################################################################################
dend_colours <- c("#0000fffd","#ff00fffd","#00fffffd","#008000fd","#ff5555fd","#25e589","#00ff00fd","#d45500fd")
library(ggradar)##
library(tidyverse)
for (c in seq(13)) {
  labels <- c("Commercial", "Estates", "High-density", "Low-density", "Open green space", "Other green", "Terraced")
  sub <- df[df$hierarchical8 == c,][,1:7]
  med <- data.frame(apply(sub, 2, median, na.rm=TRUE))
  sub_ <- t(med)
  sub_ <- cbind(a = 0, sub_)
  r <- ggradar(
    sub_,
    axis.label.size = 2.5,
    grid.label.size = 2.5,
    group.colours = c(dend_colours[c]),
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
  path <- paste0("/home/emily/phd/drives/phd/chapter4clustering/outputs/R/lsoa_hierarchical13_radar_",c,'.tex')
  tikzDevice::tikz(file = path, width = 2.5, height = 2.5)
  plot(r)
  dev.off()
  lines <- readLines(con=path)
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=path)
}
