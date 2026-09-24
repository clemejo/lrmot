import numpy as np
from scipy.linalg import solve

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds.manifold import Manifold
from pymanopt.optimizers import ConjugateGradient
from pymanopt.tools import printer
import numpy as np
import time
from copy import deepcopy

from pymanopt.optimizers import ConjugateGradient


class OTConjugateGradient(ConjugateGradient):

    def __init__(self,C,relative_cost_tolerance=1e-6,consecutive_iterations=5,max_time=180,log=False,*args,**kwargs):
        super().__init__(*args, **kwargs)

        self.C = np.asarray(C, dtype=float)
        if log:
            self.log = []
            self.is_log = True
        else:
            self.is_log = False
        self.relative_cost_tolerance = relative_cost_tolerance
        self.consecutive_iterations = consecutive_iterations
        self.max_time = max_time

    def _ot_cost(self, point):
        U, V = point
        g = U.sum(axis=0)
        return float(np.sum(U * (self.C @ V), axis=0).dot(1.0 / g))

    def run(self,problem,*,initial_point=None,reuse_line_searcher=False):
        manifold = problem.manifold
        objective = problem.cost
        gradient = problem.riemannian_gradient

        if (not reuse_line_searcher or self.line_searcher is None):
            self.line_searcher = deepcopy(self._line_searcher)

        line_searcher = self.line_searcher

        t_start = time.time()

        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point


        if self._verbosity >= 2:
            iteration_format_length = (int(np.log10(self._max_iterations)) + 1)
            column_printer = printer.ColumnPrinter(columns=[("Iteration", f"{iteration_format_length}d"),("Cost", "+.16e"),("Gradient norm", ".8e")])
        else:
            column_printer = printer.VoidPrinter()

        column_printer.print_header()

        cost = objective(x)
        previous_ot_cost = self._ot_cost(x)
        grad = gradient(x)
        gradient_norm = manifold.norm(x,grad)
        Pgrad = problem.preconditioner(x,grad)

        gradPgrad = manifold.inner_product(x,grad,Pgrad)

        descent_direction = -Pgrad

        consecutive_count = 0
        self._initialize_log(optimizer_parameters={"beta_rule": self._beta_rule,"orth_value": self._orth_value,"line_searcher": line_searcher,"relative_cost_tolerance":self.relative_cost_tolerance,"consecutive_iterations":self.consecutive_iterations,},)

        iteration = 0
        step_size = np.nan
        if self.is_log:
            self.log.append({"iteration": 0,"cost": previous_ot_cost,"elapsed_time": time.time() - t_start})

        stopping_criterion = False
        while True:

            iteration += 1
            column_printer.print_row([iteration,cost,gradient_norm])

            self._add_log_entry(iteration=iteration,point=x,cost=cost,gradient_norm=gradient_norm)

            if stopping_criterion:
                break

            df0 = manifold.inner_product(x,grad,descent_direction)

            if df0 >= 0:

                descent_direction = -Pgrad
                df0 = -gradPgrad

            step_size, newx = line_searcher.search(objective,manifold,x,descent_direction,cost,df0)
            newcost = objective(newx)

            new_ot_cost = self._ot_cost(newx)

            relative_change = (abs(new_ot_cost - previous_ot_cost) / max(abs(previous_ot_cost), 1e-15))

            if self.is_log:
                self.log.append({"iteration": iteration,"cost": new_ot_cost,"elapsed_time": time.time() - t_start})

            if relative_change < self.relative_cost_tolerance:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if self._verbosity >= 3:
                print(f"OT stopping criterion: relative change = {relative_change:.3e}, consecutive = {consecutive_count}/{self.consecutive_iterations}")

            if consecutive_count >= self.consecutive_iterations:
                stopping_criterion = True

            total_time = time.time() - t_start

            if total_time >= self.max_time:
                stopping_criterion=True

                if self._verbosity >= 2:
                    print(stopping_criterion)
                    print("")

                x = newx
                cost = newcost

                return self._return_result(
                    start_time=t_start,
                    point=x,
                    cost=cost,
                    iterations=iteration,
                    stopping_criterion=stopping_criterion,
                    cost_evaluations=iteration,
                    step_size=step_size,
                    gradient_norm=manifold.norm(x,gradient(x)))

            previous_ot_cost = new_ot_cost
            newgrad = gradient(newx)
            newgradient_norm = manifold.norm(newx,newgrad)

            Pnewgrad = problem.preconditioner(newx,newgrad)

            newgradPnewgrad = manifold.inner_product(newx,newgrad,Pnewgrad)

            oldgrad = manifold.transport(newx,grad)

            orth_grads = (manifold.inner_product(newx,oldgrad,Pnewgrad) / newgradPnewgrad)

            if abs(orth_grads) >= self._orth_value:
                beta = 0
                descent_direction = -Pnewgrad
            else:
                descent_direction = manifold.transport(newx,descent_direction)

                beta = self._beta_update(
                    manifold=manifold,
                    x=x,
                    newx=newx,
                    grad=grad,
                    newgrad=newgrad,
                    Pnewgrad=Pnewgrad,
                    newgradPnewgrad=newgradPnewgrad,
                    Pgrad=Pgrad,
                    gradPgrad=gradPgrad,
                    gradient_norm=gradient_norm,
                    oldgrad=oldgrad,
                    descent_direction=descent_direction)

                descent_direction = -Pnewgrad + beta * descent_direction

            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradient_norm = newgradient_norm
            gradPgrad = newgradPnewgrad

        return self._return_result(
            start_time=t_start,
            point=x,
            cost=cost,
            iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=iteration,
            step_size=step_size,
            gradient_norm=gradient_norm)


class BalancedLowRankOT(Manifold):

    def __init__(self,a,b,r,projection_tol=1e-12,projection_maxiter=50,retraction_tol=1e-12,retraction_maxiter=1000):
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)

        self.m = len(a)
        self.n = len(b)

        self.projection_tol = projection_tol
        self.projection_maxiter = projection_maxiter

        self.retraction_tol = retraction_tol
        self.retraction_maxiter = retraction_maxiter

        self.r = r

        super().__init__(name="Balanced low-rank OT",dimension=(len(a) + len(b) - 1) * (r - 1),point_layout=2)
        
        # PyManopt uses this to unpack (U,V) when evaluating functions
        self._point_layout = (1, 1)

    @property
    def dim(self):
        # dim(M_bal) = (m+n-1)(r-1)
        return (self.m + self.n - 1) * (self.r - 1)

    @property
    def point_layout(self):
        return self._point_layout

    def set_rank(self, r):
        self.r = r

    # Fisher-Rao metric
    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        U, V = point
        U += np.ones_like(U)*10**-12
        V += np.ones_like(V)*10**-12
        AU, AV = tangent_vector_a.U, tangent_vector_a.V
        BU, BV = tangent_vector_b.U, tangent_vector_b.V

        return float(np.sum(AU * BU / U) + np.sum(AV * BV / V))

    def norm(self, point, tangent_vector):
        return np.sqrt(max(self.inner_product(point, tangent_vector, tangent_vector),0.0))

    def zero_vector(self, point):
        U, V = point
        return BalancedTangent(np.zeros_like(U),np.zeros_like(V))

    def random_point(self):
        # Dirichlet initialization followed by KL projection onto M_bal
        U = np.random.dirichlet(np.ones(self.r), size=self.m) * self.a[:, None]
        V = np.random.dirichlet(np.ones(self.r), size=self.n) * self.b[:, None]
        return self._kl_project_balanced(U, V)

    def random_tangent_vector(self, point):
        U, V = point.U, point.V

        AU = np.random.randn(*U.shape) * U
        AV = np.random.randn(*V.shape) * V

        AU, AV = self.projection(point, (AU, AV))

        nrm = self.norm(point, (AU, AV))

        if nrm < 1e-15:
            return self.zero_vector(point)

        return BalancedTangent(AU / nrm,AV / nrm)

    def projection(self, point, vector):
        """
        Paper Eq 7:
            Pi(Z)_U = Z_U - U * (alpha 1^T + 1 gamma^T)
            Pi(Z)_V = Z_V - V * (beta 1^T - 1 gamma^T)
        where alpha, beta, gamma solve Eq 8
        """
        U, V = point
        ZU, ZV = vector.U, vector.V

        _, r = U.shape

        self.r = r

        # g = U^T 1 = V^T 1
        s = U.sum(axis=0)

        # Right-hand sides of Eq 8
        rhs_u = ZU.sum(axis=1)              
        rhs_v = ZV.sum(axis=1)              
        rhs_g = ZU.sum(axis=0) - ZV.sum(axis=0)

        # D_a^{-1} rhs_u and D_b^{-1} rhs_v.
        da_inv_rhs = rhs_u / self.a
        db_inv_rhs = rhs_v / self.b

        S = (2.0 * np.diag(s) - U.T @ (U / self.a[:, None]) - V.T @ (V / self.b[:, None]))

        schur_rhs = (rhs_g- U.T @ da_inv_rhs + V.T @ db_inv_rhs)

        # Fix gamma[-1] = 0.
        if r == 1:
            gamma = np.zeros(1)
        else:
            Sred = S[:-1, :-1]
            bred = schur_rhs[:-1]
            gamma = np.zeros(r)
            gamma[:-1] = solve(Sred,bred,assume_a="sym",check_finite=False)
            gamma[-1] = 0.0

        alpha = (rhs_u - U @ gamma) / self.a

        beta = (rhs_v + V @ gamma) / self.b

        # Eq 7
        PU = ZU - U * (alpha[:, None]+ gamma[None, :])
        PV = ZV - V * (beta[:, None] - gamma[None, :])
        return BalancedTangent(PU, PV)

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """
        Fisher-Rao gradient: grad f = Pi(U * grad_U f, V * grad_V f)
        Paper Eq 12
        """
        U, V = point
        GU, GV = euclidean_gradient
        return self.projection(point,BalancedTangent(U * GU, V * GV))

    def retraction(self, point, tangent_vector):
        """
        Retraction from Eq 10 + Algorithm 1
        """
        U, V = point
        U += np.ones_like(U)*10**-12
        V += np.ones_like(V)*10**-12
        XiU, XiV = tangent_vector.U, tangent_vector.V

        Ubar = U * np.exp(np.clip(XiU / U, -50.0, 50.0))
        Vbar = V * np.exp(np.clip(XiV / V, -50.0, 50.0))
        return self._kl_project_balanced(Ubar, Vbar)

    def _kl_project_balanced(self, U, V):
        # Algorithm 1 from the paper
        U = np.maximum(U, np.finfo(float).tiny)
        V = np.maximum(V, np.finfo(float).tiny)

        for _ in range(self.retraction_maxiter):

            U *= (self.a / U.sum(axis=1))[:, None]
            V *= (self.b / V.sum(axis=1))[:, None]

            sU = U.sum(axis=0)
            sV = V.sum(axis=0)

            t = np.sqrt(sV / sU)

            U *= t[None, :]
            V /= t[None, :]

            err_u = np.max(np.abs(U.sum(axis=1) - self.a))
            err_v = np.max(np.abs(V.sum(axis=1) - self.b))
            err_s = np.max(np.abs(U.sum(axis=0) - V.sum(axis=0)))

            if max(err_u, err_v, err_s) < self.retraction_tol:
                break

        return U, V

    def transport(self, point_b, tangent_vector_a):
        # A valid vector transport is obtained by projecting the old tangent vector onto the tangent space at the new point
        return self.projection(point_b,tangent_vector_a)

class BalancedTangent:

    __slots__ = ("U", "V")

    def __init__(self, U, V):
        self.U = U
        self.V = V

    def __neg__(self):
        return BalancedTangent(-self.U, -self.V)

    def __add__(self, other):
        return BalancedTangent(self.U + other.U,self.V + other.V)

    def __sub__(self, other):
        return BalancedTangent(self.U - other.U,self.V - other.V)

    def __mul__(self, scalar):
        return BalancedTangent(scalar * self.U,scalar * self.V)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return BalancedTangent(self.U / scalar,self.V / scalar)

    def __repr__(self):
        return (f"BalancedTangent(U={self.U.shape}, V={self.V.shape})")

class BalancedLinearOT:

    def __init__(self, C):
        self.C = np.asarray(C, dtype=float)

    def cost(self, U, V):
        # f(U,V) = sum_k (u_k^T C v_k) / g_k
        g = U.sum(axis=0)

        CV = self.C @ V

        numerator = np.sum(U * CV, axis=0)

        return float(np.sum(numerator / g))

    def euclidean_gradient(self, U, V):
        g = U.sum(axis=0)

        CV = self.C @ V

        numerator = np.sum(U * CV, axis=0)

        GU = CV / g[None, :] - np.ones_like(U) * (numerator / (g * g))[None, :]
        GVT = self.C.T @ U
        GV = GVT / g[None, :]
        return GU, GV

    def coupling(self, U, V):
        # Construct the dense coupling
        g = U.sum(axis=0)
        return U @ np.diag(1.0 / g) @ V.T

def make_initial_point(a, b, r, seed=0):
    """
    Initialization described in Appendix B.1 of the paper:
    Dirichlet rows followed by balanced KL projection
    """
    rng = np.random.default_rng(seed)
    U = rng.dirichlet(np.ones(r), size=len(a)) * np.asarray(a)[:, None]
    V = rng.dirichlet(np.ones(r), size=len(b)) * np.asarray(b)[:, None]    
    return U, V


def solve_balanced_low_rank_ot(C,a,b,rank,max_iterations=1000,tol=1e-6,max_time=180,min_gradient_norm=1e-8,verbosity=2,beta_rule="HestenesStiefel",projection_tol=1e-12,retraction_tol=1e-12,seed=0,log=False):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    np.random.seed(seed)

    manifold = BalancedLowRankOT(a,b,rank,projection_tol=projection_tol,retraction_tol=retraction_tol)
    manifold.set_rank(rank)

    ot = BalancedLinearOT(C)

    @pymanopt.function.numpy(manifold)
    def cost(U, V):
        return ot.cost(U, V)

    @pymanopt.function.numpy(manifold)
    def egrad(U, V):
        return ot.euclidean_gradient(U, V)

    problem = Problem(manifold,cost,euclidean_gradient=egrad)

    optimizer = OTConjugateGradient(
        C=C,
        beta_rule=beta_rule,
        max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
        verbosity=verbosity,
        relative_cost_tolerance=tol,
        consecutive_iterations=5,
        max_time=max_time,
        log=log)

    U0, V0 = make_initial_point(a, b, rank, seed=seed)
    U0, V0 = manifold._kl_project_balanced(U0, V0)

    result = optimizer.run(problem,initial_point=(U0, V0))

    if log:
        return result, manifold, optimizer.log
    else:
        return result, manifold


if __name__ == "__main__":

    np.random.seed(123)

    m = 30
    n = 35
    r = 5

    X = np.random.randn(m, 2)
    Y = np.random.randn(n, 2)

    C = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2.0 * X @ Y.T

    a = np.ones(m) / m
    b = np.ones(n) / n

    result, manifold, log = solve_balanced_low_rank_ot(C=C,a=a,b=b,rank=r,max_iterations=1000,min_gradient_norm=1e-6,beta_rule="HestenesStiefel",verbosity=2,seed=42,log=True)

    U, V = result.point

    print("\nOptimization finished.")
    print("Final cost:", result.cost)
    print("Iterations:", result.iterations)
    print("Gradient norm:", result.gradient_norm)

    row_error_U = np.max(np.abs(U.sum(axis=1) - a))
    row_error_V = np.max(np.abs(V.sum(axis=1) - b))
    latent_error = np.max(np.abs(U.sum(axis=0) - V.sum(axis=0)))

    print("\nFeasibility:")
    print("||U 1 - a||_inf =", row_error_U)
    print("||V 1 - b||_inf =", row_error_V)
    print("||U^T 1 - V^T 1||_inf =", latent_error)

    ot = BalancedLinearOT(C)

    Gamma = ot.coupling(U, V)

    print("\nCoupling diagnostics:")
    print("Gamma shape:", Gamma.shape)
    print("||Gamma 1 - a||_inf =",np.max(np.abs(Gamma.sum(axis=1) - a)))
    print("||Gamma^T 1 - b||_inf =",np.max(np.abs(Gamma.sum(axis=0) - b)))
    print("<C,Gamma> =",np.sum(C * Gamma))
