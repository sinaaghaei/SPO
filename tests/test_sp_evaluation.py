import unittest
import logging
import random

from solver.sp_reformulation import SpReformulation
from utils.data_generator import SpDataGenerator
from utils.sp_graph import SpGraph
from oracle.sp_oracle import SpOracle
from solver.sp_evaluation import *

logging.captureWarnings(True)  # To suppress the warnings


class MyUnitTest(unittest.TestCase):
    def test_sp_evaluation_1(self):
        """
        In this module we check the sanity of the evaluation module
        """
        # parameters
        random.seed(30)
        dim = 5
        num_covariate_features = 5
        regularization = "lasso"  # "ridge lasso
        _lambda = 0.0001
        _alpha = 0.001
        time_limit = 600

        grid = SpGraph(dim)
        sp_oracle = SpOracle(incidence_matrix=grid.A, start_node=0, end_node=24,
                             small_coefficient_tolerance=0.01,
                             time_limit=60)

        data_generator = SpDataGenerator(polykernel_degree=1, polykernel_noise_half_width=0, d=grid.n_edges,
                                         p=num_covariate_features)
        B_true = data_generator.B_true
        x_train, c_train = data_generator.generate_poly_kernel_data(n=100)
        x_test, c_test = data_generator.generate_poly_kernel_data(n=1000)
        # Add intercept in the first row of X
        x_train = data_generator.add_intercept(x_train)
        x_test = data_generator.add_intercept(x_test)

        sp_reformulation_instance = SpReformulation(x_train, c_train, sp_oracle, regularization, _lambda, _alpha,
                                                    time_limit)
        sol_dict = sp_reformulation_instance.solve_model()
        if sol_dict['status'] in ["infeasible", "inf_or_unbd"]:
            pass
        else:
            B_new = sol_dict['B_ast']
            spo_loss_value = spo_loss(B_new, x_test, c_test, sp_oracle)
            spo_plus_loss_value = spo_plus_loss(B_new, x_test, c_test, sp_oracle)
            least_square_loss_value = least_squares_loss(B_new, x_test, c_test)
            absolute_loss_value = absolute_loss(B_new, x_test, c_test)

            print("spo_loss:", spo_loss_value)
            print("spo_plus_loss:", spo_plus_loss_value)
            print("least_square_loss:", least_square_loss_value)
            print("absolute_loss:", absolute_loss_value)


if __name__ == '__main__':
    unittest.main()
