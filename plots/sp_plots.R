library(tidyverse)
library(gridExtra)

rm(list=ls())
graphics.off()
setwd("/Users/sina/Documents/GitHub/DSO-SPO/results/")

results= read.csv('sp_results_100_t_60.csv', header=TRUE, sep=',', na.strings="", stringsAsFactors =TRUE)
results$zstar_avg_test =( results$z_star_sum_test)/(results$n_test )

results <- results %>% mutate(spo_loss_SPOplus = spo_loss_SPOplus/zstar_avg_test,
                              spo_loss_LSE = spo_loss_LSE/zstar_avg_test,
                              spo_loss_SPOplusLSE = spo_loss_SPOplusLSE/zstar_avg_test)

results_spo <- results[,c("dim","num_covariate_features","n_train","n_test","kernel_degree","kernel_noise","trial","spo_loss_SPOplus","spo_loss_LSE","spo_loss_SPOplusLSE")]
results_lse <- results[,c("dim","num_covariate_features","n_train","n_test","kernel_degree","kernel_noise","trial","least_square_loss_SPOplus","least_square_loss_LSE","least_square_loss_SPOplusLSE")]

names(results_spo) = names(results_lse) = c("grid_dim","p_features","n_train","n_test","kernel_degree","kernel_noise","trial","SPO+","LSE","SPO++")


results_relevant_fixed = results_spo %>%
  gather(`SPO+`, `LSE`, `SPO++`,
         key = "method", value = "spo_normalized")

# results_relevant_fixed = results_lse %>%
#   gather(`SPO+`, `LSE`, `SPO++`,
#          key = "method", value = "LSE_loss")

results_relevant_fixed$method = as.factor(results_relevant_fixed$method)
results_relevant_fixed$n_train = as.factor(results_relevant_fixed$n_train)
results_relevant_fixed$kernel_noise = as.factor(results_relevant_fixed$kernel_noise)
results_relevant_fixed$grid_dim = as.factor(results_relevant_fixed$grid_dim)
results_relevant_fixed$p_features = as.factor(results_relevant_fixed$p_features)

# Labelers
training_set_size_names <- c(
  '100' = "Training Set Size = 100",
  '1000' = "Training Set Size = 1000",
  '5000' = "Training Set Size = 5000"
)

half_width_names <- c(
  '0' = "Noise Half-width = 0",
  '0.5' = "Noise Half-width = 0.5" 
)

p_features_names <- c(
  '5' = "p = 5",
  '10' = "p = 10"
)

grid_dim_names <- c(
  '2' = "2 x 2 grid",
  '3' = "3 x 3 grid",
  '4' = "4 x 4 grid",
  '5' = "5 x 5 grid" 
)

training_set_size_labeller <- as_labeller(training_set_size_names)
half_width_labeller <- as_labeller(half_width_names)
p_features_labeller <- as_labeller(p_features_names)
grid_dim_labeller <- as_labeller(grid_dim_names)


####### BOX PLOT ####### 

plot <- results_relevant_fixed %>%
  ggplot(aes(x = as.factor(kernel_degree), y = spo_normalized, fill = method)) + # spo_normalized LSE_loss
  geom_boxplot() +
  scale_y_continuous(name = "Normalized SPO Loss", labels = scales::percent_format(accuracy = 1)) +
  scale_fill_discrete(name = "Method") +
  facet_wrap(vars(n_train, kernel_noise), 
             labeller = labeller(n_train = training_set_size_labeller, kernel_noise = half_width_labeller), 
             ncol = 2, scales = "free") + 
  theme_bw() +
  labs(x = "Polynomial Degree", title = "Normalized SPO Loss vs. Polynomial Degree") +
  theme(axis.title=element_text(size=36), axis.text=element_text(size=30), legend.text=element_text(size=36), 
        legend.title=element_text(size=36), strip.text = element_text(size = 24), 
        legend.position="top", plot.title = element_text(size = 42, hjust = 0.5))

plot

ggsave("shotest_path_plot_SPO.pdf", width = 20, height = 18, units = "in")
