import unittest
import logging
import random
import numpy as np

from solver.sp_reformulation import SpReformulation
from utils.data_generator import SpDataGenerator
from utils.sp_graph import SpGraph
from oracle.sp_oracle import SpOracle

logging.captureWarnings(True)  # To suppress the warnings


class MyUnitTest(unittest.TestCase):
    def test_sp_calibrator_1(self):
        """
        In this module we check the sanity of the reformulation approach for the shortest path problem
        """
        # parameters
        random.seed(30)
        dim = 5
        num_covariate_features = 5
        regularization = "lasso"  # "ridge lasso
        time_limit = 600
        n_train = 100
        n_test = 10000
        holdout_percent = 0.25
        n_holdout = np.floor(holdout_percent * n_train)
        num_lambda = 10
        lambda_max = 100
        lambda_min_ratio = 10.0 ^ (-8)

        grid = SpGraph(dim)
        sp_oracle = SpOracle(incidence_matrix=grid.A, start_node=0, end_node=24,
                             small_coefficient_tolerance=0.01,
                             time_limit=60)

        data_generator = SpDataGenerator(polykernel_degree=1, polykernel_noise_half_width=0, d=grid.n_edges,
                                         p=num_covariate_features)
        B_true = data_generator.B_true
        x_train, c_train = data_generator.generate_poly_kernel_data(n=n_train)
        x_test, c_test = data_generator.generate_poly_kernel_data(n=n_test)
        x_calibration, c_calibration = data_generator.generate_poly_kernel_data(n=n_holdout)
        # Add intercept in the first row of X
        x_train = data_generator.add_intercept(x_train)
        x_test = data_generator.add_intercept(x_test)
        x_calibration = data_generator.add_intercept(x_calibration)



if __name__ == '__main__':
    unittest.main()