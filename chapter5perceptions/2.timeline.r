library(tidyverse)
library(scales)
library(ggplot2)
library(grid)
library(reshape2)
library(hrbrthemes)
source('/home/emily/phd/drives/phd/ggplot_theme_Publication/ggplot_theme_Publication-2.R')

# install.packages("extrafont")
# library(extrafont)
# # Install **TTF** Latin Modern Roman fonts from www.fontsquirrel.com/fonts/latin-modern-roman
# # Import the newly installed LModern fonts, change the pattern according to the 
# # filename of the lmodern ttf files in your fonts folder
font_import(pattern = "latinmodernroman_10regular_macroman/lmroman*")


library(extrafont)
loadfonts(device = "all")
par(family = "LM Roman 10")
df <- read.csv('/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.csv')

g1 <- ggplot(df, aes(x = as.Date(date), y = idx)) +
  geom_point(size = 1, color="#999999") +
  xlab("Date") +
  ylab("Cumulative Counts") +
  #scale_fill_hue(c = 40)  + 
  #scale_colour_Publication() + 
  theme_Publication() + ylim(-60,28000) +
  theme(text=element_text(size=16,  family="LM Roman 10")) 
g1 <- g1 + theme(text = element_text(size =  40),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 40),
                 axis.line = element_line(colour = 'black', size = 2)
                 )
g1
png(filename="/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.png", width = 900, height=600)
plot(g1)
dev.off()

tikzDevice::tikz(file = "/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.tex", width = 3.5, height = 3, standAlone=TRUE)

# here we add a LaTeX title to the plot
g1 <- g1 + theme(text = element_text(size =  6),
                 axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)), 
                 axis.title = element_text(size = 6))
                 #,axis.text.x = element_text(angle = 45)

g1
# closing the graphics device saves the file we opened with tikzDevice::tikz
dev.off()

tools::texi2dvi("/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.tex",pdf=T)
system(paste(getOption('pdfviewer'),file.path(g1,'/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.pdf')))

path <- "/home/emily/phd/drives/phd/chapter5perceptions/outputs/R/timeline.tex"
lines <- readLines(con=path)
lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
writeLines(lines,con=path)