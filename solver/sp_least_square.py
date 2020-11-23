'''
This module solves the empirical risk least squares problem formulated as follows

    Min_{B} 2norm(Bx-c)^2 + _lambda omega(B)

The solver outputs the B* and corresponding hyper-parameter _lambda
'''

from gurobipy import *
import numpy as np


class SpLeastSquare:
    def __init__(self, x_train, c_train, oracle, regularization, _lambda, time_limit,
                 regularize_first_column_B=False,
                 upper_bound_B_present=False, upper_bound_B=10.0 ** 6):
        self.x = x_train
        self.c = c_train
        self.oracle = oracle
        self.regularization = regularization
        self._lambda = _lambda
        self.time_limit = time_limit
        self.regularize_first_column_B = regularize_first_column_B
        self.upper_bound_B_present = upper_bound_B_present
        self.upper_bound_B = upper_bound_B

    def solve_model(self):
        '''
        This function create the least sqaure problem and solve it using gurobi
        :return: the optimal solution B and corresponding hyper-parameter _lambda
        '''
        # dimension check
        p, n = self.x.shape
        d, n_c = self.c.shape
        if n != n_c:
            raise Exception("Error: Dimensions of the input are mismatched.")

        model = Model('sp_least_square')
        model.params.TimeLimit = self.time_limit
        model.params.OutputFlag = 1

        if self.upper_bound_B_present:
            B_var = model.addVars(d, p, vtype=GRB.CONTINUOUS, lb=-1 * self.upper_bound_B, ub=self.upper_bound_B,
                                  name="B")
        else:
            B_var = model.addVars(d, p, vtype=GRB.CONTINUOUS, name="B")

        obj = QuadExpr(0)

        '''
        The MSE loss is |Bx - c|^2 = sum_{i \in {1,..,n}} sum_{j \in {1,..,d}} [(sum_{k \in {1,...,p}}B_jk*x_ki)-c_ji]^2

        we define auxiliary decision variable beta_ij = (sum_{k \in {1,...,p}}B_jk*x_ki)-c_ji

        so that the loss become \sum_i \sum_j (beta_ij * beta_ij)
        '''

        beta_var = model.addVars(n, d, vtype=GRB.CONTINUOUS, name="beta")
        model.addConstrs(
            (quicksum(B_var[j, k] * self.x[k, j] for k in range(p)) - self.c[j, i] == beta_var[i, j]) for i in
            range(n) for j in range(d))
        obj.add(quicksum(beta_var[i, j] * beta_var[i, j] for i in range(n) for j in range(d)))

        if self.regularization == "lasso":
            theta_var = model.addVars(d, p, vtype=GRB.CONTINUOUS, name="theta")  # theta_var = abs(B_var)
            model.addConstrs(theta_var[i, j] >= B_var[i, j] for i in range(d) for j in range(p))
            model.addConstrs(theta_var[i, j] >= -B_var[i, j] for i in range(d) for j in range(p))

        # Add regularization part for the decision variable B
        if self.regularization == "ridge" and self.regularize_first_column_B:
            obj.add(n * self._lambda * quicksum(B_var[i, j] * B_var[i, j] for i in range(d) for j in range(p)))
        elif self.regularization == "ridge" and (not self.regularize_first_column_B):
            obj.add(n * self._lambda * quicksum(B_var[i, j] * B_var[i, j] for i in range(d) for j in range(1, p)))
        elif self.regularization == "lasso" and self.regularize_first_column_B:
            obj.add(2 * n * self._lambda * quicksum(theta_var[i, j] for i in range(d) for j in range(p)))
        elif self.regularization == "lasso" and (not self.regularize_first_column_B):
            obj.add(2 * n * self._lambda * quicksum(theta_var[i, j] for i in range(d) for j in range(1, p)))
        else:
            raise Exception("enter valid regularization: :ridge or :lasso")

        model.setObjective(obj, GRB.MINIMIZE)

        model.update()
        model.optimize()
        sol = dict()
        status = model.getAttr("Status")
        obj_value = None
        B_ast = np.zeros((d, p))
        if status == 2:
            status = "optimal"
            obj_value = model.getAttr("ObjVal")
            for i in range(d):
                B_ast[i,] = [B_var[i, j].X for j in range(p)]
        elif status == 9:
            status = "timelimit reached"
            obj_value = model.getAttr("ObjVal")
            for i in range(d):
                B_ast[i,] = [B_var[i, j].X for j in range(p)]
        elif status == 3:
            status = "infeasible"
        elif status == 4:
            status = "inf_or_unbd"

        sol['lambda'] = self._lambda
        sol['obj_value'] = obj_value
        sol['status'] = status
        sol['B_ast'] = B_ast

        return sol
