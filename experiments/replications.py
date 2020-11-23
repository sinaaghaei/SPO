'''
In this module
'''

from utils.data_generator import SpDataGenerator
from solver.sp_calibrator import SpCalibrator
from solver.sp_evaluation import *


class SpReplication:
    def __init__(self, n_train, n_test, holdout_percent, kernel_degree, kernel_noise, grid, sp_oracle,
                 num_covariate_features, regularization, time_limit, num_lambda, lambda_max, lambda_min_ratio,
                 num_alpha, alpha_max, alpha_min_ratio):
        self.n_train = n_train
        self.n_test = n_test
        self.holdout_percent = holdout_percent
        self.n_holdout = int(self.holdout_percent * self.n_train)
        self.holdout_percent = holdout_percent
        self.kernel_degree = kernel_degree
        self.kernel_noise = kernel_noise
        self.grid = grid
        self.oracle = sp_oracle
        self.num_covariate_features = num_covariate_features
        self.regularization = regularization
        self.time_limit = time_limit
        self.num_lambda = num_lambda
        self.lambda_max = lambda_max
        self.lambda_min_ratio = lambda_min_ratio
        self.num_alpha = num_alpha
        self.alpha_max = alpha_max
        self.alpha_min_ratio = alpha_min_ratio

    def replicate(self):
        # Here we generate the training, test and calibration data
        data_generator = SpDataGenerator(polykernel_degree=self.kernel_degree,
                                         polykernel_noise_half_width=self.kernel_noise,
                                         d=self.grid.n_edges,
                                         p=self.num_covariate_features)

        B_true = data_generator.B_true
        x_train, c_train = data_generator.generate_poly_kernel_data(n=self.n_train)
        x_test, c_test = data_generator.generate_poly_kernel_data(n=self.n_test)
        x_calibration, c_calibration = data_generator.generate_poly_kernel_data(n=self.n_holdout)
        # Add intercept in the first row of X
        x_train = data_generator.add_intercept(x_train)
        x_test = data_generator.add_intercept(x_test)
        x_calibration = data_generator.add_intercept(x_calibration)

        # Solving LSE, SPO_plus and SPO_plus_LSE approaches along with calibration lambda and alpha
        sp_calibrator_instance = SpCalibrator(x_train, c_train, x_calibration, c_calibration, self.oracle,
                                              self.regularization,
                                              self.time_limit, self.num_lambda,
                                              self.lambda_max, self.lambda_min_ratio, self.num_alpha,
                                              self.alpha_max, self.alpha_min_ratio)

        best_B_SPOplus, best_lambda_SPOplus, _, _, cal_avg_gap_SPOplus, cal_num_of_inf_or_unbd_SPOplus = sp_calibrator_instance.calibrate(
            "SPO_plus")
        best_B_LSE, best_lambda_LSE, _, _, cal_avg_gap_LSE, cal_num_of_inf_or_unbd_LSE = sp_calibrator_instance.calibrate(
            "LSE")
        best_B_SPOplus_LSE, best_lambda_SPOplus_LSE, best_alpha_SPOplus_LSE, _, cal_avg_gap_SPOplus_LSE, cal_num_of_inf_or_unbd_SPOplus_LSE = sp_calibrator_instance.calibrate(
            "SPO_plus_LSE")

        spo_loss_SPOplus = spo_loss(best_B_SPOplus, x_test, c_test, self.oracle)
        spo_loss_LSE = spo_loss(best_B_LSE, x_test, c_test, self.oracle)
        spo_loss_SPOplusLSE = spo_loss(best_B_SPOplus_LSE, x_test, c_test, self.oracle)
        least_square_loss_SPOplus = least_squares_loss(best_B_SPOplus, x_test, c_test)
        least_square_loss_LSE = least_squares_loss(best_B_LSE, x_test, c_test)
        least_square_loss_SPOplusLSE = least_squares_loss(best_B_SPOplus_LSE, x_test, c_test)
        results_tmp = [best_lambda_SPOplus, None, spo_loss_SPOplus, least_square_loss_SPOplus, cal_avg_gap_SPOplus,
                       cal_num_of_inf_or_unbd_SPOplus,
                       best_lambda_LSE, None, spo_loss_LSE, least_square_loss_LSE, cal_avg_gap_LSE,
                       cal_num_of_inf_or_unbd_LSE,
                       best_lambda_SPOplus_LSE, best_alpha_SPOplus_LSE, spo_loss_SPOplusLSE,
                       least_square_loss_SPOplusLSE, cal_avg_gap_SPOplus_LSE, cal_num_of_inf_or_unbd_SPOplus_LSE]

        return results_tmp
