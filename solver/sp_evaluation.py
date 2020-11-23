'''
In this module for a given B_ast produced by one of the approaches and a evaluation data
(x,c) we ouput various loss values:
    1- least_squares_loss
    2- spo_loss
    3- spo_plus_loss
    4- absolute_loss
'''

import numpy as np

def least_squares_loss(B_new, x, c):
    p, n = x.shape
    residuals = B_new@x - c
    error = (1 / n) * (1 / 2) * (np.linalg.norm(residuals) ** 2)
    return error


def spo_loss(B_new, x, c, oracle, z_star=None):
    spo_loss = 0
    if z_star is None:
        z_star, _ = oracle.oracle_dataset(c)
    n = len(z_star)
    for i in range(n):
        c_hat_i = B_new @ x[:, i]
        sol_i = oracle.solve_oracle(c_hat_i)
        w_oracle_i = sol_i['w_ast']
        spo_loss_i = c[:, i] @ w_oracle_i - z_star[i]
        spo_loss += spo_loss_i

    spo_loss_avg = spo_loss / n
    return spo_loss_avg


def spo_plus_loss(B_new, x, c, oracle, z_star=None, w_star=None):
    if z_star is None or w_star is None:
        z_star, w_star = oracle.oracle_dataset(c)

    spo_plus_sum = 0
    n = len(z_star)

    for i in range(n):
        c_hat_i = B_new @ x[:, i]
        spoplus_cost_vec = 2 * c_hat_i - c[:, i]
        sol_i = oracle.solve_oracle(spoplus_cost_vec)
        z_oracle_i = sol_i['z_ast']
        spo_plus_cost = -z_oracle_i + 2 * c_hat_i @ w_star[:, i] - z_star[i]
        spo_plus_sum = spo_plus_sum + spo_plus_cost

    spo_plus_avg = spo_plus_sum / n
    return spo_plus_avg


def absolute_loss(B_new, x, c):
    p, n = x.shape
    residuals = B_new @ x - c
    error = (1 / n) * np.sum(abs(residuals))

    return error
