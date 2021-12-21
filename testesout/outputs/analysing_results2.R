
library(readr)
library(viridis)
library(dplyr)
library(ggplot2)
library(ggpubr)


import_rename <- function(dataname, bias, noise){
  
  df1 <- read_csv(paste0("testesout/outputs/datasets_accuracy/", dataname, '.csv') )
  df1$model[df1$model == 'error_classic'] <- 'RWCN'
  df1$model[df1$model == 'error_classic_bin'] <- 'BWCN'
  df1$model[df1$model == 'error_encoding'] <- 'RWQN'
  df1$model[df1$model == 'error_HSGS'] <- 'BWQN'
  
  
  mean_error = aggregate(df1$error, by=list(df1$model), mean)
  sd_error = aggregate(df1$error, by=list(df1$model), sd)
  results = cbind(mean_error, sd_error[,2])
  colnames(results) = c('model', 'avg_error', 'std_error')
  
  results$bias = bias
  results$noise = noise
  
  
  return(results)
}




#=============
# DATASET 2
#=============

x1 = import_rename('dataset2_experiments_bias_1noise', 'biased', '1 noise')
y1 = import_rename('dataset2_experiments_original_1noise', 'unbiased', '1 noise')
x2 = import_rename('dataset2_experiments_bias_2noises', 'biased', '2 noises')
y2 = import_rename('dataset2_experiments_original_2noises', 'unbiased', '2 noises')
x3 = import_rename('dataset2_experiments_bias_3noises', 'biased', '3 noises')
y3 = import_rename('dataset2_experiments_original_3noises', 'unbiased', '3 noises')

data2 = rbind(x1, y1, x2, y2, x3, y3)
data2 = reshape(data2, idvar = c("model", 'bias'), timevar = "noise", direction = "wide")
write.csv(data2, 'testesout/outputs/tables/dataset2_experiments.csv',  row.names = F)

# DATASET 3
#=====

x1 = import_rename('dataset3_experiments_bias_1noise', 'biased', '1 noise')
y1 = import_rename('dataset3_experiments_original_1noise', 'unbiased', '1 noise')
x2 = import_rename('dataset3_experiments_bias_2noises', 'biased', '2 noises')
y2 = import_rename('dataset3_experiments_original_2noises', 'unbiased', '2 noises')
x3 = import_rename('dataset3_experiments_bias_3noises', 'biased', '3 noises')
y3 = import_rename('dataset3_experiments_original_3noises', 'unbiased', '3 noises')

data3 = rbind(x1, y1, x2, y2, x3, y3)
data3 = reshape(data3, idvar = c("model", 'bias'), timevar = "noise", direction = "wide")
write.csv(data3, 'testesout/outputs/tables/dataset3_experiments.csv',  row.names = F)


# DATASET 4
#=====

x1 = import_rename('dataset4_experiments_bias_1noise', 'biased', '1 noise')
y1 = import_rename('dataset4_experiments_original_1noise', 'unbiased', '1 noise')
x2 = import_rename('dataset4_experiments_bias_2noises', 'biased', '2 noises')
y2 = import_rename('dataset4_experiments_original_2noises', 'unbiased', '2 noises')
x3 = import_rename('dataset4_experiments_bias_3noises', 'biased', '3 noises')
y3 = import_rename('dataset4_experiments_original_3noises', 'unbiased', '3 noises')

data4 = rbind(x1, y1, x2, y2, x3, y3)
data4 = reshape(data4, idvar = c("model", 'bias'), timevar = "noise", direction = "wide")
write.csv(data4, 'testesout/outputs/tables/dataset4_experiments.csv',  row.names = F)



# DATASET 5
#=====


x1 = import_rename('dataset5_experiments_bias_1noise', 'biased', '1 noise')
y1 = import_rename('dataset5_experiments_original_1noise', 'unbiased', '1 noise')
x2 = import_rename('dataset5_experiments_bias_2noises', 'biased', '2 noises')
y2 = import_rename('dataset5_experiments_original_2noises', 'unbiased', '2 noises')
x3 = import_rename('dataset5_experiments_bias_3noises', 'biased', '3 noises')
y3 = import_rename('dataset5_experiments_original_3noises', 'unbiased', '3 noises')

data5 = rbind(x1, y1, x2, y2, x3, y3)
data5 = reshape(data5, idvar = c("model", 'bias'), timevar = "noise", direction = "wide")
write.csv(data5, 'testesout/outputs/tables/dataset5_experiments.csv',  row.names = F)

