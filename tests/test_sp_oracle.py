import unittest
import logging
import numpy as np
from utils.sp_graph import SpGraph
from oracle.sp_oracle import SpOracle

logging.captureWarnings(True)  # To suppress the warnings


class MyUnitTest(unittest.TestCase):
    def test_sp_oracle_1(self):
        """
        We test if the shortest path oracle works properly
        """
        test_sp_oracle = SpOracle(incidence_matrix=SpGraph(20).A, start_node=0, end_node=399,
                                  small_coefficient_tolerance=0.01,
                                  time_limit=60)
        c_test = np.ones(760)
        result = test_sp_oracle.solve_oracle(c_test)
        self.assertEqual(result['z_ast'], 38)

    def test_sp_oracle_2(self):
        """
        We test if the shortest path oracle works properly
        """
        A = np.array([[-1, -1, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 1, 1]])
        test_sp_oracle = SpOracle(incidence_matrix=A, start_node=0, end_node=3, small_coefficient_tolerance=0.01,
                                  time_limit=60)
        c_test = np.array([4.0, 1, 4, 6])
        result = test_sp_oracle.solve_oracle(c_test)
        self.assertEqual(result['z_ast'], 7)
        np.testing.assert_array_equal(result['w_ast'], np.array([0, 1.0, 0, 1.0]))


if __name__ == '__main__':
    unittest.main()
