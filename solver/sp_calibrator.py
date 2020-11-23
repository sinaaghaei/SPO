'''
In this module, we calibrate the hyper parameters _lambda and _alpha for a given instance.
We create a sequence of lambdas. For each value in the sequence we train the model on the training set and evaluate
spo_loss on the calibration set. We return the one with the lowest loss on the calibration set.

For calibrating alpha, we first calibrate lambda and then given the best lambda, similarly we find the best alpha.
'''

from solver.sp_least_square import SpLeastSquare
from solver.sp_reformulation import SpReformulation
from solver.sp_evaluation import *

from gurobipy import *
import numpy as np


class SpCalibrator:
    def __init__(self, x_train, c_train, x_calibration, c_calibration, oracle, regularization, time_limit, num_lambda,
                 lambda_max, lambda_min_ratio, num_alpha,
                 alpha_max, alpha_min_ratio, loss_metric="SPO_loss", acyclic=False,
                 regularize_first_column_B=False,
                 upper_bound_B_present=False, upper_bound_B=10.0 ** 6):
        self.x_train = x_train
        self.c_train = c_train
        self.x_calibration = x_calibration
        self.c_calibration = c_calibration

        self.oracle = oracle
        self.regularization = regularization
        self.time_limit = time_limit
        self.acyclic = acyclic
        self.regularize_first_column_B = regularize_first_column_B
        self.upper_bound_B_present = upper_bound_B_present
        self.upper_bound_B = upper_bound_B
        self.num_lambda = num_lambda
        self.lambda_max = lambda_max
        self.lambda_min_ratio = lambda_min_ratio
        self.num_alpha = num_alpha
        self.alpha_max = alpha_max
        self.alpha_min_ratio = alpha_min_ratio
        self.loss_metric = loss_metric

        # Constructing the lambda sequence
        self.lambda_sequence = None
        if self.num_lambda == 1 and self.lambda_max == 0:
            lambda_sequence = [0.0]
        else:
            lambda_min = self.lambda_max * self.lambda_min_ratio
            gap = np.log(self.lambda_max) - np.log(lambda_min)
            log_lambdas = np.arange(np.log(lambda_min), np.log(self.lambda_max), gap / self.num_lambda)
            self.lambda_sequence = np.exp(log_lambdas)

        # Constructing the alpha sequence
        self.alpha_sequence = self.lambda_sequence

    def calibrate(self, method):
        '''

        :param method: Name of the approach we want to calibrate
        :return: Matrix B corresponding to the lambda and alpha with the lowest loss on calibration set
                The best lambda and alpha
                The average optimality gap for instances we solve per values of lambda and alpha
                number of instances that got infeasible or unbounded
        '''
        best_alpha = None
        if method == "LSE":
            best_B, best_lambda, best_loss, avg_gap, num_of_inf_or_unbd = self.calibrate_LSE()
        elif method == "SPO_plus":
            best_B, best_lambda, best_loss, avg_gap, num_of_inf_or_unbd = self.calibrate_SPO_plus()
        elif method == "SPO_plus_LSE":
            best_B, best_lambda, best_alpha, best_loss, avg_gap, num_of_inf_or_unbd = self.calibrate_SPO_plus_LSE()
        else:
            raise Exception("Wrong method name")

        return best_B, best_lambda, best_alpha, best_loss, avg_gap, num_of_inf_or_unbd

    def calibrate_LSE(self):
        best_B = None
        best_lambda = None
        best_loss = np.PINF
        sum_gap = 0
        num_of_inf_or_unbd = 0
        num_of_feasible = 0
        for lambda_cur in self.lambda_sequence:
            sp_least_square_instance = SpLeastSquare(self.x_train, self.c_train, self.oracle, self.regularization,
                                                     lambda_cur, self.time_limit)
            sol_dict = sp_least_square_instance.solve_model()
            if sol_dict['status'] in ["infeasible", "inf_or_unbd"]:
                num_of_inf_or_unbd += 1

            else:
                B_new = sol_dict['B_ast']
                sum_gap += sol_dict['gap']
                num_of_feasible += 1
                if self.loss_metric == "SPO_loss":
                    spo_loss_value = spo_loss(B_new, self.x_calibration, self.c_calibration, self.oracle)
                    if spo_loss_value < best_loss:
                        best_loss = spo_loss_value
                        best_B = B_new
                        best_lambda = lambda_cur

        return best_B, best_lambda, best_loss, sum_gap / num_of_feasible, num_of_inf_or_unbd

    def calibrate_SPO_plus(self):
        best_B = None
        best_lambda = None
        best_loss = np.PINF
        sum_gap = 0
        num_of_inf_or_unbd = 0
        num_of_feasible = 0
        for lambda_cur in self.lambda_sequence:
            sp_reformulation_instance = SpReformulation(self.x_train, self.c_train, self.oracle, self.regularization,
                                                        lambda_cur, 0, self.time_limit)  # alpha = 0
            sol_dict = sp_reformulation_instance.solve_model()
            if sol_dict['status'] in ["infeasible", "inf_or_unbd"]:
                num_of_inf_or_unbd += 1
            else:
                B_new = sol_dict['B_ast']
                sum_gap += sol_dict['gap']
                num_of_feasible += 1
                if self.loss_metric == "SPO_loss":
                    spo_loss_value = spo_loss(B_new, self.x_calibration, self.c_calibration, self.oracle)
                    if spo_loss_value < best_loss:
                        best_loss = spo_loss_value
                        best_B = B_new
                        best_lambda = lambda_cur

        return best_B, best_lambda, best_loss, sum_gap / num_of_feasible, num_of_inf_or_unbd

    def calibrate_SPO_plus_LSE(self):
        '''
        We first find the best lambda without including alpha regularizer in the objective by setting alpha to 0
        and then given the best lambda we calibrate alpha.

        :return: best_B, best_lambda, best_alpha
        '''
        _, best_lambda, _, _, _ = self.calibrate_SPO_plus()
        best_B = None
        best_alpha = None
        best_loss = np.PINF
        sum_gap = 0
        num_of_inf_or_unbd = 0
        num_of_feasible = 0
        for alpha_cur in self.alpha_sequence:
            sp_reformulation_instance = SpReformulation(self.x_train, self.c_train, self.oracle, self.regularization,
                                                        best_lambda, alpha_cur, self.time_limit)
            sol_dict = sp_reformulation_instance.solve_model()
            if sol_dict['status'] in ["infeasible", "inf_or_unbd"]:
                num_of_inf_or_unbd += 1
            else:
                B_new = sol_dict['B_ast']
                sum_gap += sol_dict['gap']
                num_of_feasible += 1
                if self.loss_metric == "SPO_loss":
                    spo_loss_value = spo_loss(B_new, self.x_calibration, self.c_calibration, self.oracle)
                    if spo_loss_value < best_loss:
                        best_loss = spo_loss_value
                        best_B = B_new
                        best_alpha = alpha_cur

        return best_B, best_lambda, best_alpha, best_loss, sum_gap / num_of_feasible, num_of_inf_or_unbd
