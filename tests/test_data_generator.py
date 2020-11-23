import unittest
import logging
import numpy as np
from utils.data_generator import SpDataGenerator

logging.captureWarnings(True)  # To suppress the warnings


class MyUnitTest(unittest.TestCase):
    def test_sp_data_generator_1(self):
        """
        We test the outputted data; no assertions
        """
        dg = SpDataGenerator(polykernel_degree=1, polykernel_noise_half_width=0, d=2, p=3)
        x_tmp, c_tmp = dg.generate_poly_kernel_data(n=5)
        print("x_temp\n",x_tmp)
        print("c_temp\n", c_tmp)


if __name__ == '__main__':
    unittest.main()
