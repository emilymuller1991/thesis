# deprivation perception scatter 
library(ggplot2)
library(colorspace)

df_perceptions <-  read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/deprivation_merge_perceptions_norm.csv')

# df <- tidyr::pivot_longer(df_perceptions, c(safety, wealth, boring, depressing, lively, walk, beauty), 
#                           names_to = "perception", values_to = "perception_scores")
# df <- tidyr::pivot_longer(df, c(Living.Environment.Score,
#                                 Crime.Score, Health.Deprivation.and.Disability.Score,
#                                 Barriers.to.Housing.and.Services.Score,
#                                 Education..Skills.and.Training.Score, Employment.Score..rate.,
#                                 Income.Score..rate., Index.of.Multiple.Deprivation..IMD..Score,
#                                 Outdoors.Sub.domain.Score),
#                           names_to = "deprivation", values_to = "deprivation_scores")
# 
# p <- ggplot(df, aes(perception_scores, deprivation_scores)) +
#   geom_smooth(method='lm') +
#   stat_cor() +
#   theme_Publication()  +
#   geom_point(size=0.01)
# 
# p + facet_grid(perception ~ deprivation)

df <-  read.csv('/home/emily/phd/drives/phd/chapter6substantive/outputs/deprivation_merge_perceptions_corr.csv')
dfm <- melt(df[,colnames(df)],id.vars = 1)
# Give exereme colors:
g1 <- ggplot(dfm, aes(as.factor(variable), X, fill= value)) + 
  geom_tile() +
  scale_fill_distiller(palette = "RdBu", direction=1) +
  xlab("Perception Scores") +
  ylab("Deprivation Scores") +
  #ggtitle('2011') +
  theme_Publication() +
  coord_fixed() +
  geom_text(aes(label=round(value,2)), size =  2.75) +
  scale_x_discrete(labels= c('Safety', 'Wealth', 'Boring', 'Depressing', 'Lively', 'Walk', 'Beauty'))
g1

g1 <- g1 + theme(text = element_text(size =  3),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 6),
                 axis.text.x = element_text(size = 6, angle=45, vjust = 0.99, hjust=1),
                 axis.text.y = element_text(size = 6),
                 axis.line = element_line(size=0),
                 legend.position = "none",
                 legend.direction = "vertical",
                 legend.title	= element_text(size=0)
)
g1

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter6substantive/outputs/R/correlation_deprivation_perceptions.tex", width = 5, height = 5)
g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()
path <- "/home/emily/phd/drives/phd/chapter6substantive/outputs/R/correlation_deprivation_perceptions.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)

# png(filename="/home/emily/phd/drives/phd/chapter6substantive/outputs/scatterplot_deprivation_perceptions_res.png", width = 700, height=700)
# plot(g1)
# dev.off()

