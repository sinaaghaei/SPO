'''
In this class we formulate a shortest path problem in the form of following linear optimization :

                                z^\ast(c) := min c^Tw
                                            s.t. Aw >= b

The solution of the above problem returned by the oracle is referred to as w^\ast(c) in
the paper.

A specific implementation of an optimization oracle should take a single input argument c,
and should return the pair (z^\ast(c), w^\ast(c))

This file contains the code for the shortest path oracle
'''

from gurobipy import *
import numpy as np


class SpOracle:
    def __init__(self, incidence_matrix, start_node, end_node, small_coefficient_tolerance, time_limit):
        self.A_mat = incidence_matrix
        self.n_nodes, self.n_edges = self.A_mat.shape
        self.start_node = start_node
        self.end_node = end_node
        self.small_coefficient_tolerance = small_coefficient_tolerance
        self.time_limit = time_limit

        # Set up RHS
        self.b_vec = np.zeros(self.n_nodes)
        self.b_vec[self.start_node] = -1
        self.b_vec[self.end_node] = 1

        # decision variables
        self.w = None

    def solve_oracle(self, c):
        model = Model('sp_oracle')
        model.params.TimeLimit = self.time_limit
        model.params.OutputFlag = 0

        self.w = model.addMVar(self.n_edges, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        model.addMConstrs(self.A_mat, self.w, '=', self.b_vec)

        d = len(c)
        for i in range(d):
            if abs(c[i]) < self.small_coefficient_tolerance:
                c[i] = 0

        model.setObjective(c @ self.w, GRB.MINIMIZE)
        # model.setMObjective(None, c, 0.0, None, None, self.w, GRB.MINIMIZE)
        model.update()
        model.optimize()

        z_ast = model.getAttr("ObjVal")
        w_ast = self.w.X

        sol = dict()

        sol['z_ast'] = z_ast
        sol['w_ast'] = w_ast

        return sol

    def oracle_dataset(self, c):
        '''
        :return: Given matrix c \in R^(d*n) this function call the oracle for each c[:,i] i \in {1,..,n}
        and return the corresponding z*(c_i) and w*(c_i)
        '''
        d, n = c.shape
        z_star_data = np.zeros(n)
        w_star_data = np.zeros((d, n))

        for i in range(n):
            sol_i = self.solve_oracle(c[:, i])
            z_star_data[i] = sol_i['z_ast']
            w_star_data[:, i] = sol_i['w_ast']

        return z_star_data, w_star_data
