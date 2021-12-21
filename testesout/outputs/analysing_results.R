
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)


boxplotResults <- function(dataname, xname, yname, title=''){
  
  df1 <- read_csv(paste0("testesout/outputs/datasets_accuracy/", dataname, '.csv') )
  df1$model[df1$model == 'error_classic'] <- 'classic_real'
  df1$model[df1$model == 'error_classic_bin'] <- 'classic_binary'
  df1$model[df1$model == 'error_encoding'] <- 'quantum_real'
  df1$model[df1$model == 'error_HSGS'] <- 'quantum_binary'
  
plot = ggplot(data=df1, aes(x=model, y=error, fill=model)) +
        geom_boxplot() +
        #scale_fill_viridis( alpha=0.6) +
        scale_y_continuous(limits = c(0, 0.5))+
        #geom_jitter(color="black", size=0.4, alpha=0.9) +
        theme_minimal() +
        theme(
          legend.position="none",
          plot.title = element_text(size=11, face='bold'),
          axis.text.x = element_text(size=7,
                                     angle = 12, 
                                     hjust=0.5,
                                     vjust=1,
                                     face="plain")
        ) +
        labs(x=xname, y=yname, title=title)


  return(plot)
}

#=============
# DATASET 2
#=============

x1 = boxplotResults('dataset2_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset2_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset2_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset2_experiments_original_2noises', 'without bias', '', '2 noises')


x3 = boxplotResults('dataset2_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset2_experiments_original_3noises', 'without bias', '', '3 noises')

ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset2_experiments.png'), height = 6, width = 6, units = 'in')



# DATASET 3
#=====

x1 = boxplotResults('dataset3_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset3_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset3_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset3_experiments_original_2noises', 'without bias', '', '2 noises')


x3 = boxplotResults('dataset3_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset3_experiments_original_3noises', 'without bias', '', '3 noises')

ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset3_experiments.png'), height = 6, width = 6, units = 'in')


# DATASET 4
#=====

x1 = boxplotResults('dataset4_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset4_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset4_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset4_experiments_original_2noises', 'without bias', '', '2 noises')


x3 = boxplotResults('dataset4_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset4_experiments_original_3noises', 'without bias', '', '3 noises')

ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset4_experiments.png'), height = 6, width = 6, units = 'in')


# DATASET 5
#=====

x1 = boxplotResults('dataset5_experiments_bias_1noise', 'with bias', 'error')
y1 = boxplotResults('dataset5_experiments_original_1noise', 'without bias', '', '1 noise')

x2 = boxplotResults('dataset5_experiments_bias_2noises', 'with bias', 'error')
y2 = boxplotResults('dataset5_experiments_original_2noises', 'without bias', '', '2 noises')

x3 = boxplotResults('dataset5_experiments_bias_3noises', 'with bias', 'error')
y3 = boxplotResults('dataset5_experiments_original_3noises', 'without bias', '', '3 noises')

ggarrange(y1, x1, y2, x2, y3, x3, ncol = 2, nrow = 3)
ggsave(paste0('Mestrado/qneuronreal/testesout/outputs/plots/dataset5_experiments.png'), height = 6, width = 6, units = 'in')

