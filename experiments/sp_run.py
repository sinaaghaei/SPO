import logging
import random
import pandas as pd

from experiments.replications import SpReplication
from utils.sp_graph import SpGraph
from oracle.sp_oracle import SpOracle

logging.captureWarnings(True)  # To suppress the warnings


def main():
    # parameters
    random.seed(30)
    dim = 5
    num_covariate_features = 5
    regularization = "lasso"  # "ridge lasso
    time_limit = 120
    n_test = 10000
    holdout_percent = 0.25
    num_lambda = num_alpha = 10
    lambda_max = alpha_max = 100
    lambda_min_ratio = alpha_min_ratio = 10.0 ** (-8)

    n_train_vec = [100]
    polykernel_degree_vec = [1, 2, 4, 6, 8]
    polykernel_noise_half_width_vec = [0, 0.5]

    grid = SpGraph(dim)
    sp_oracle = SpOracle(incidence_matrix=grid.A, start_node=0, end_node=(grid.n_nodes - 1), time_limit=time_limit)

    # Here we create an empty dataframe for the results. In the following loop we insert rows per iteration
    results_dataframe = pd.DataFrame(
        columns=["dim", "num_covariate_features", "regularization", "time_limit", "n_train", "n_test",
                 "holdout_percent",
                 "kernel_degree", "kernel_noise", "num_lambda", "num_alpha", "trial",
                 "best_lambda_SPOplus", "best_alpha_SPOplus", "spo_loss_SPOplus", "least_square_loss_SPOplus",
                 "cal_avg_gap_SPOplus", "cal_num_of_inf_or_unbd_SPOplus",
                 "best_lambda_LSE", "best_alpha_LSE", "spo_loss_LSE", "least_square_loss_LSE", "cal_avg_gap_LSE",
                 "cal_num_of_inf_or_unbd_LSE",
                 "best_lambda_SPOplus_LSE", "best_alpha_SPOplus_LSE", "spo_loss_SPOplusLSE",
                 "least_square_loss_SPOplusLSE", "cal_avg_gap_SPOplus_LSE", "cal_num_of_inf_or_unbd_SPOplus_LSE"])

    for n_train in n_train_vec:
        for kernel_degree in polykernel_degree_vec:
            for kernel_noise in polykernel_noise_half_width_vec:
                for trial in range(5):
                    print(
                        "####################### n_train: {}, polykernel_degree:{}, noise:{}".format(n_train,
                                                                                                     kernel_degree,
                                                                                                     kernel_noise))
                    row_items = [dim, num_covariate_features, regularization, time_limit, n_train, n_test,
                                 holdout_percent,
                                 kernel_degree, kernel_noise, num_lambda, num_alpha, trial]
                    #
                    sp_replicator = SpReplication(n_train, n_test, holdout_percent, kernel_degree, kernel_noise, grid,
                                                  sp_oracle, num_covariate_features, regularization, time_limit,
                                                  num_lambda, lambda_max, lambda_min_ratio,
                                                  num_alpha, alpha_max, alpha_min_ratio)
                    results_tmp = sp_replicator.replicate()
                    row_items = row_items + results_tmp
                    new_row = pd.DataFrame([row_items], columns=list(results_dataframe.columns))
                    results_dataframe = results_dataframe.append(new_row, ignore_index=True)

    results_dataframe.to_csv('./../results/sp_results_100.csv', index=False)


if __name__ == "__main__":
    main()
