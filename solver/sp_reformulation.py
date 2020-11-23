'''
Solves the empirical risk SPO+ problem for the specific case of the shortest path problem
with the reformulation approach given in formulation (17) in the paper. We have also added a second regularization term
which is the MSE loss of the predictions penalized with parameter __alpha in the objective

The feasible region S in the nominal problem is S = {w: Aw = b}. Thus, the reformulation would be as follows:

                            Min_{B,p} 1\n \sum_{i=1}^{n} {-b_vec^T p_i + 2 (w_star(c_i) x_i^T).B - z_star(c_i) + _lambda omega(B) + _alpha MSE(c_i - Bx_i)

                            s.t. A^T p_i - 2Bx_i + c_i - s_var <= 0 \forall i in


The solver outputs the B* and corresponding hyper-parameters _lambda and _alpha
'''

from gurobipy import *
import numpy as np


class SpReformulation:
    def __init__(self, x_train, c_train, oracle, regularization, _lambda, _alpha, time_limit, acyclic=False,
                 regularize_first_column_B=False,
                 upper_bound_B_present=False, upper_bound_B=10.0 ** 6):
        self.x = x_train
        self.c = c_train
        self.oracle = oracle
        self.regularization = regularization
        self._lambda = _lambda
        self._alpha = _alpha
        self.time_limit = time_limit
        self.acyclic = acyclic
        self.regularize_first_column_B = regularize_first_column_B
        self.upper_bound_B_present = upper_bound_B_present
        self.upper_bound_B = upper_bound_B

    def solve_model(self):
        '''
        This function create the problem in formulation (17) and solve it using gurobi
        :return: the optimal solution B and corresponding hyper-parameters _lambda and _alpha
        '''
        # dimension check
        p, n = self.x.shape
        d, n_c = self.c.shape
        if n != n_c:
            raise Exception("Error: Dimensions of the input are mismatched.")

        A_mat = self.oracle.A_mat
        b_vec = self.oracle.b_vec
        n_nodes, n_edges = A_mat.shape
        if n_edges != d:
            raise Exception("Error: Dimensions of the input are mismatched.")

        z_star_data, w_star_data = self.oracle.oracle_dataset(self.c)  # We get all the solutions of the oracle apriori

        model = Model('sp_reformulation')
        model.params.TimeLimit = self.time_limit
        model.params.OutputFlag = 0

        # Here we add the variables to the model
        p_var = model.addVars(n_nodes, n, vtype=GRB.CONTINUOUS, lb=0, name="p")
        if not self.acyclic:
            s_var = model.addVars(d, n, vtype=GRB.CONTINUOUS, lb=0, name="s")

        if self.upper_bound_B_present:
            B_var = model.addVars(d, p, vtype=GRB.CONTINUOUS, lb=-1 * self.upper_bound_B, ub=self.upper_bound_B,
                                  name="B")
        else:
            B_var = model.addVars(d, p, vtype=GRB.CONTINUOUS, name="B")

        # Here we add the constraints to the model

        # \foreach i A^Tp_i -2Bx_i + c_i - s_i
        if self.acyclic:
            model.addConstrs((quicksum(A_mat.T[j, k] * p_var[k, i] for k in range(n_nodes)) - 2 * quicksum(
                self.x[k, i] * B_var[j, k] for k in range(p)) + self.c[j, i] <= 0) for i in range(n) for j in
                             range(d))
        else:
            model.addConstrs((quicksum(A_mat.T[j, k] * p_var[k, i] for k in range(n_nodes)) - 2 * quicksum(
                self.x[k, i] * B_var[j, k] for k in range(p)) + self.c[j, i] - s_var[j, i] <= 0) for i in range(n) for j
                             in
                             range(d))

        if self.regularization == "lasso":
            theta_var = model.addVars(d, p, vtype=GRB.CONTINUOUS, name="theta")  # theta_var = abs(B_var)
            model.addConstrs(theta_var[i, j] >= B_var[i, j] for i in range(d) for j in range(p))
            model.addConstrs(theta_var[i, j] >= -B_var[i, j] for i in range(d) for j in range(p))

        # Here we add the objective function
        if self.regularization == "ridge":
            obj = QuadExpr(0)
        elif self.regularization == "lasso" and self._alpha == 0:
            obj = LinExpr(0)
        else:
            obj = QuadExpr(0)

        # obj_noreg = \sum_{i} -b^Tp_i + 2 w_ast_i @ B_var@x_i - z_ast_i + 1^Ts_i
        for i in range(n):
            if self.acyclic:
                cur_expr = -1 * quicksum(b_vec[k] * p_var[k, i] for k in range(n_nodes)) - z_star_data[i]
                for j in range(p):
                    cur_expr += 2 * self.x[j, i] * quicksum(w_star_data[k, i] * B_var[k, j] for k in range(d))
            else:
                cur_expr = -1 * quicksum(b_vec[k] * p_var[k, i] for k in range(n_nodes)) + quicksum(
                    s_var[k, i] for k in range(d)) - z_star_data[i]
                for j in range(p):
                    cur_expr += 2 * self.x[j, i] * quicksum(w_star_data[k, i] * B_var[k, j] for k in range(d))
            obj.add(cur_expr)

        # Add regularization part for the decision variable B
        if self.regularization == "ridge" and self.regularize_first_column_B:
            obj.add(n * (self._lambda / 2) * quicksum(B_var[i, j] * B_var[i, j] for i in range(d) for j in range(p)))
        elif self.regularization == "ridge" and (not self.regularize_first_column_B):
            obj.add(n * (self._lambda / 2) * quicksum(B_var[i, j] * B_var[i, j] for i in range(d) for j in range(1, p)))
        elif self.regularization == "lasso" and self.regularize_first_column_B:
            obj.add(n * self._lambda * quicksum(theta_var[i, j] for i in range(d) for j in range(p)))
        elif self.regularization == "lasso" and (not self.regularize_first_column_B):
            obj.add(n * self._lambda * quicksum(theta_var[i, j] for i in range(d) for j in range(1, p)))
        else:
            raise Exception("enter valid regularization: :ridge or :lasso")

        # Add the MSE loss regularization
        '''
        The MSE loss is |Bx - c|^2 = sum_{i \in {1,..,n}} sum_{j \in {1,..,d}} [(sum_{k \in {1,...,p}}B_jk*x_ki)-c_ji]^2
        
        we define auxiliary decision variable beta_ij = (sum_{k \in {1,...,p}}B_jk*x_ki)-c_ji
        
        so that the loss become \sum_i \sum_j (beta_ij * beta_ij)
        '''
        if self._alpha != 0:
            beta_var = model.addVars(n, d, vtype=GRB.CONTINUOUS, name="beta")
            model.addConstrs(
                (quicksum(B_var[j, k] * self.x[k, j] for k in range(p)) - self.c[j, i] == beta_var[i, j]) for i in
                range(n) for j in range(d))
            obj.add((self._alpha / 2) * quicksum(beta_var[i, j] * beta_var[i, j] for i in range(n) for j in range(d)))

        model.setObjective(obj, GRB.MINIMIZE)

        model.update()
        model.optimize()
        sol = dict()
        status = model.getAttr("Status")
        obj_value = None
        gap = 100
        B_ast = np.zeros((d, p))
        if status == 2:
            status = "optimal"
            gap = 0
            obj_value = model.getAttr("ObjVal")
            for i in range(d):
                B_ast[i,] = [B_var[i, j].X for j in range(p)]
        elif status == 9:
            status = "timelimit reached"
            obj_value = model.getAttr("ObjVal")
            gap = model.getAttr("MIPGap") * 100
            if gap > 100:
                gap = 100
            for i in range(d):
                B_ast[i,] = [B_var[i, j].X for j in range(p)]
        elif status == 3:
            status = "infeasible"
        elif status == 4:
            status = "inf_or_unbd"

        sol['lambda'] = self._lambda
        sol['alpha'] = self._alpha
        sol['obj_value'] = obj_value
        sol['status'] = status
        sol['B_ast'] = B_ast
        sol['gap'] = gap

        return sol
