import numpy as np
from model import ml_train_test_random_cross_validation_states

x_month_inadvance = 1#lead time is one month
seed_num=5#number of random cross validation experiments
x_all_save_raw=f'x_all_{x_month_inadvance}'#model inputs
y_all_save_raw=f'y_all_{x_month_inadvance}'#model inputs
him_dim=4#dimension of the hidden state vector in FF module
para_dir=f"./"#direcotry for model parameters to save/load
path_FF=f"{him_dim}_{x_month_inadvance}_monthly"
para_path_FF = para_dir+path_FF#model parameters for the FF module
para_path_npp=para_dir+f"npp_limitation"#model parameters for the FA module
para_path_popu=para_dir+f"hs_limitation"#model parameters for the HS module
####train, validate, and evaluate the ML fire model
result_metrics=ml_train_test_random_cross_validation_states\
    (seed_num,x_all_save_raw,y_all_save_raw,para_path_FF,para_path_npp,para_path_popu,him_dim)
####print the mean and std of the F1-score
print(f'F1-score', np.mean(result_metrics[:, 1]), np.std(result_metrics[:, 1]))
