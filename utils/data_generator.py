'''
    In this module we want to generate historical data including of n datapoints (x_i, c_i) where x_i \in R^p
    and c_i \in R^d. This module would return matrices
        x_observed \in R^(p*n)
        c_observed \in R^(d*n)


    Please refer to the paper for the details.

    In this module we have  a class "SpDataGenerator" where in the instructor we pass the required parameters
    for generating data (X,c). In the constrcuter we build the true matrix B. Then each time we want to generate data, we call
    function "generate_poly_kernel_data". We should pass number of datapoints as the input.
'''
import numpy as np


class SpDataGenerator:
    def __init__(self, polykernel_degree, polykernel_noise_half_width, d, p, normalize_c=False,
                 kernel_damp_normalize=False, inner_constant=3,
                 outer_constant=1, normalize_small_threshold=0.0001):
        self.polykernel_degree = polykernel_degree
        self.noise_half_width = polykernel_noise_half_width
        self.d = d
        self.p = p
        self.normalize_c = normalize_c
        self.kernel_damp_normalize = kernel_damp_normalize
        self.inner_constant = inner_constant
        self.outer_constant = outer_constant
        self.normalize_small_threshold = normalize_small_threshold
        self.B_true = np.random.binomial(1, 0.5, (self.d, self.p))
        # self.B_true = np.random.randn(self.d, self.p)

    def generate_poly_kernel_data(self, n):
        '''

        :param n: the size of the dataset
        :return: x_observed the matrix of the covariate features with dimension (p,n)
                 c_observed the matrix of the weights with dimension (d,n)
        '''
        kernel_damp_factor = 1 / np.sqrt(self.p)

        x_observed = np.random.randn(self.p, n)
        dot_prods = self.B_true @ x_observed

        dot_prods = dot_prods * kernel_damp_factor
        if self.kernel_damp_normalize:
            dot_prods = dot_prods / np.linalg.norm(self.B_true, axis=1).reshape(-1, 1)

        c_observed = (dot_prods + self.inner_constant) ** self.polykernel_degree + self.outer_constant
        noise = (1 - self.noise_half_width) + 2 * self.noise_half_width * np.random.rand(*dot_prods.shape)

        c_observed = c_observed * noise

        if self.normalize_c:
            c_observed = c_observed / np.linalg.norm(c_observed, axis=0)
            c_observed[np.abs(c_observed) < self.normalize_small_threshold] = 0

        # x : (p,n)  c: (d,n)
        return x_observed, c_observed

    def add_intercept(self, x):
        '''
        This function add a row with all entries being one to the input matrix x
        :param x: covaiate data
        :return: covariate data with added intercept feature
        '''
        intercept = np.ones(x.shape[1]).reshape(1,-1)
        x_new = np.append(intercept, x, 0)
        return x_new