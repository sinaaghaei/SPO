import unittest
import logging
import random

from solver.sp_least_square import SpLeastSquare
from utils.data_generator import SpDataGenerator
from utils.sp_graph import SpGraph
from oracle.sp_oracle import SpOracle

logging.captureWarnings(True)  # To suppress the warnings


class MyUnitTest(unittest.TestCase):
    def test_sp_least_square_1(self):
        """
        In this module we check the sanity of the least square approach for the shortest path problem
        """
        # parameters
        random.seed(30)
        dim = 5
        num_covariate_features = 5
        regularization = "lasso"  # "ridge lasso
        _lambda = 0.0001
        time_limit = 600

        grid = SpGraph(dim)
        sp_oracle = SpOracle(incidence_matrix=grid.A, start_node=0, end_node=24,
                             small_coefficient_tolerance=0.01,
                             time_limit=60)

        data_generator = SpDataGenerator(polykernel_degree=1, polykernel_noise_half_width=0, d=grid.n_edges,
                                         p=num_covariate_features)
        B_true = data_generator.B_true
        x_train, c_train = data_generator.generate_poly_kernel_data(n=100)
        # Add intercept in the first row of X
        x_train = data_generator.add_intercept(x_train)

        sp_least_square_instance = SpLeastSquare(x_train, c_train, sp_oracle, regularization, _lambda, time_limit)
        sol_dict = sp_least_square_instance.solve_model()
        print(sol_dict['status'])
        print(sol_dict['obj_value'])
        print(sol_dict['B_ast'])
        print(sol_dict['B_ast'].shape)


if __name__ == '__main__':
    unittest.main()
