import numpy as np
from utils import *


def project_C1(Qs, g, a_list, alpha, eps=1e-300):
    Qs_proj = []

    for Q, a in zip(Qs, a_list):
        row_sums = np.sum(Q, axis=1)
        row_sums = np.maximum(row_sums, eps)
        Q_proj = Q * (a / row_sums)[:, None]
        Qs_proj.append(Q_proj)

    g_proj = np.maximum(g, alpha)
    return Qs_proj, g_proj


def project_C2(Qs, g, eps=1e-300):
    d = len(Qs)

    col_sums = [np.maximum(np.sum(Q, axis=0), eps) for Q in Qs]

    # log(g*) = [log(g) + sum_k log(colsum_k)] / (d+1)
    log_g_star = (np.log(np.maximum(g, eps)) + sum(np.log(cs) for cs in col_sums)) / (d + 1)
    g_star = np.exp(log_g_star)

    Qs_proj = [Q * (g_star / cs)[None, :] for Q, cs in zip(Qs, col_sums)]

    return Qs_proj, g_star


def make_dykstra_projector(a_list, alpha=1e-12, tol=1e-8, max_iter=1000, eps=1e-300, verbose=False):
    # KL projection onto C1 \cap C2 with Dykstra

    def projector(Qs_init, g_init):
        Qs = [Q.copy() for Q in Qs_init]
        g = g_init.copy()

        p1_Q = [np.ones_like(Q) for Q in Qs]
        p1_g = np.ones_like(g)

        p2_Q = [np.ones_like(Q) for Q in Qs]
        p2_g = np.ones_like(g)

        for it in range(max_iter):

            # Projection onto C1
            Y_Q = [Q * p for Q, p in zip(Qs, p1_Q)]
            Y_g = g * p1_g
            Z_Q, Z_g = project_C1(Y_Q,Y_g,a_list,alpha,eps=eps)

            # Dykstra correction for C1
            p1_Q = [Y / np.maximum(Z, eps) for Y, Z in zip(Y_Q, Z_Q)]
            p1_g = Y_g / np.maximum(Z_g, eps)

            # Projection onto C2
            Y_Q = [Z * p for Z, p in zip(Z_Q, p2_Q)]
            Y_g = Z_g * p2_g
            X_Q, X_g = project_C2(Y_Q,Y_g,eps=eps)

            # Dykstra correction for C2
            p2_Q = [Y / np.maximum(X, eps) for Y, X in zip(Y_Q, X_Q)]
            p2_g = Y_g / np.maximum(X_g, eps)

            # Convergence
            change = max(max(np.max(np.abs(X - Q)) for X, Q in zip(X_Q, Qs)),np.max(np.abs(X_g - g)))

            Qs = X_Q
            g = X_g

            if change < tol:
                break

        return Qs, g

    return projector


def compute_gradient(C, Qs, g, C_scale, eps=1e-300):
    # gradient of f

    d = len(Qs)
    r = len(g)

    # p^(k) = Q^(k) / g
    safe_g = np.maximum(g, eps)
    Ps = [Q / safe_g[None, :] for Q in Qs]

    grad_Qs = []
    for k in range(d):
        grad_k = np.zeros_like(Qs[k])
        for s in range(r):
            tensor = C
            remaining_axes = list(range(d))
            for l in range(d):
                if l == k:
                    continue

                axis = remaining_axes.index(l)
                tensor = np.tensordot(tensor,Ps[l][:, s],axes=([axis], [0]))
                remaining_axes.pop(axis)

            grad_k[:, s] = tensor

        grad_Qs.append(grad_k / C_scale)

    c = np.zeros(r)
    for s in range(r):
        tensor = C
        remaining_axes = list(range(d))
        for k in range(d):
            axis = remaining_axes.index(k)
            tensor = np.tensordot(tensor,Ps[k][:, s],axes=([axis], [0]))
            remaining_axes.pop(axis)

        c[s] = tensor
    grad_g = -(d - 1) * c / C_scale
    return grad_Qs, grad_g

def compute_gradient_pairwise(Xs, Qs, g, C_scale, eps=1e-300):
    #using pairwise cost

    d = len(Xs)

    safe_g = np.maximum(g, eps)

    Ps = [Q / safe_g[None, :] for Q in Qs]

    Xnorm2 = [np.sum(X * X, axis=1) for X in Xs]

    # means[k][:, s] = sum_i p_{i,s}^(k) x_i^(k)
    means = [X.T @ P for X, P in zip(Xs, Ps)]

    # norm_means[k][s] = sum_i p_{i,s}^(k) \|x_i^(k)\|^2
    norm_means = [Xnorm2[k] @ Ps[k] for k in range(d)]

    # Sums over l:
    # total_mean[:,s] = sum_l m_l,s
    # total_norm[s] = sum_l q_l,s
    total_mean = sum(means)
    total_norm = sum(norm_means)

    grad_Qs = []

    for k in range(d):

        # For every s:
        # u_k,s = (d-1)\|x_i^k\|^2+ sum_{l \ne k} q_l,s - 2 x_i^k^\top sum_{l \ne k} m_l,s
        mean_excluding_k = total_mean - means[k]
        norm_excluding_k = total_norm - norm_means[k]

        # X_k @ mean_excluding_k gives n_k * r
        cross_term = Xs[k] @ mean_excluding_k

        grad_k = ((d - 1) * Xnorm2[k][:, None] + norm_excluding_k[None, :] - 2.0 * cross_term)

        grad_Qs.append(grad_k / C_scale)

    # c_s = sum_{k<l} [q_k + q_l - 2 m_k^\top m_l]
    # First two terms: (d-1) sum_k q_k
    # For the cross terms: sum_{k<l} m_k^\top m_l = 1/2 [ \|sum_k m_k\|^2 - sum_k \|m_k\|^2 ]

    cross_sum = np.sum(total_mean * total_mean, axis=0)

    for k in range(d):
        cross_sum -= np.sum(means[k] * means[k], axis=0)

    c = ((d - 1) * total_norm - cross_sum)

    grad_g = -(d - 1) * c / C_scale

    return grad_Qs, grad_g

def initialize_feasible(a_list, r, seed=42):
    # random feasible initialization

    rng = np.random.default_rng(seed)
    g = np.abs(rng.random(r)) + 1
    g = g / np.sum(g)

    Qs = []

    for a in a_list:
        ns = len(a)
        Q = np.abs(rng.random((ns, r)))
        Q = (Q.T * (a / np.sum(Q, axis=1))).T
        Qs.append(Q)

    return Qs, g

def impcp_mirror_descent(C,a_list,reg,r,alpha=1e-8,eta=0.1,dykstra_tol=1e-8,dykstra_max_iter=1000,max_iter=1000,tol=1e-6,C_scale=None,verbose=False,return_history=False):
    # for general costs

    C = np.asarray(C, dtype=float)
    if C_scale is None:
        C_scale = C.max() # may fail if C is too big 
    Qs, g = initialize_feasible(a_list,r)

    projector = make_dykstra_projector(a_list=a_list,alpha=alpha,tol=dykstra_tol,max_iter=dykstra_max_iter,verbose=False)

    history = {"objective": [],"f": [],"regularized_objective": [],"step_size": [],"change": []}
    previous_objective = None

    for t in range(max_iter):

        grad_Qs, grad_g = compute_gradient(C,Qs,g,C_scale)

        exponent = 1.0 - eta * reg
        K_Qs = []
        for Q, grad_Q in zip(Qs, grad_Qs):
            log_K = (exponent * np.log(np.maximum(Q, 1e-300)) - eta * grad_Q)
            log_K = np.clip(log_K, -700.0, 700.0)
            K = np.exp(log_K)
            K_Qs.append(K)


        log_Kg = (exponent * np.log(np.maximum(g, 1e-300)) - eta * grad_g)
        log_Kg = np.clip(log_Kg, -700.0, 700.0)
        K_g = np.exp(log_Kg)
        Qs_new, g_new = projector(K_Qs, K_g)

        change = max(max(np.max(np.abs(Qnew - Qold)) for Qnew, Qold in zip(Qs_new, Qs)),np.max(np.abs(g_new - g)))

        Qs = Qs_new
        g = g_new

        f_value = compute_objective_lr(C, Qs, g, C_scale)
        objective = f_value - reg*entropy_h(Qs,g)

        history["objective"].append(f_value)
        history["f"].append(f_value)
        history["regularized_objective"].append(objective)
        history["step_size"].append(eta)
        history["change"].append(change)

        if verbose and (t % 10 == 0 or t == max_iter - 1):
            print(f"iter={t:5d} | f={f_value:.8e} | f-eps*h={objective:.8e} | change={change:.3e}")

        if previous_objective is not None:
            objective_change = abs(objective - previous_objective) / max(1.0, abs(previous_objective))
            if change < tol and objective_change < tol:
                if verbose:
                    print(f"Mirror Descent converged at iteration {t}.")
                break
        previous_objective = objective

    if return_history:
        return Qs, g, history
    else:
        return Qs, g


def impcp_mirror_descent_pairwise(Xs, a_list, reg, r, alpha=1e-8, eta=0.1, dykstra_tol=1e-8, dykstra_max_iter=1000, max_iter=1000, tol=1e-6, seed=42, C_scale=None, verbose=False, log=False):
    # using pairwise cost

    np = ot.backend.get_backend(a_list[0])
    Qs, g = initialize_feasible(a_list,r,seed)

    projector = make_dykstra_projector(a_list=a_list,alpha=alpha,tol=dykstra_tol,max_iter=dykstra_max_iter,verbose=False)
    if log:
        log = {"objective": [],"f": [],"regularized_objective": [],"step_size": [],"change": [],"cpt": 0}

    if C_scale is None:
        C_scale = get_C_scale(Xs) # not exact but faster

    prev_loss = 42
    for t in range(max_iter):

        grad_Qs, grad_g = compute_gradient_pairwise(Xs, Qs, g, C_scale)

        exponent = 1.0 - eta * reg

        K_Qs = []
        for Q, grad_Q in zip(Qs, grad_Qs):
            log_K = (exponent * np.log(np.maximum(Q, 1e-300)) - eta * grad_Q)
            log_K = np.clip(log_K, -700.0, 700.0)
            K = np.exp(log_K)
            K_Qs.append(K)


        log_Kg = (exponent * np.log(np.maximum(g, 1e-300)) - eta * grad_g)
        log_Kg = np.clip(log_Kg, -700.0, 700.0)

        K_g = np.exp(log_Kg)
        Qs_new, g_new = projector(K_Qs, K_g)

        # Convergence
        change = max(max(np.max(np.abs(Qnew - Qold)) for Qnew, Qold in zip(Qs_new, Qs)), np.max(np.abs(g_new - g)))

        Qs = Qs_new
        g = g_new

        f_value = compute_objective_pairwise(Xs, Qs, g, C_scale)
        objective = f_value - reg*entropy_h(Qs,g)
        if log:
            log["objective"].append(f_value)
            log["f"].append(f_value)
            log["regularized_objective"].append(objective)
            log["step_size"].append(eta)
            log["change"].append(change)
            log["cpt"] = t

        if verbose and (t % 10 == 0 or t == max_iter - 1):
            print(f"iter={t:5d} | f={f_value:.8e} | f-eps*h={objective:.8e} | change={change:.3e}")

        curr_loss = objective
        err = abs(curr_loss - prev_loss) / abs(prev_loss) if prev_loss != 0.0 else np.nan

        if err < tol:
            if verbose:
                print(f"Mirror Descent converged at iteration {t}.")
            break
    
        prev_loss = curr_loss

    if log:
        return Qs, g, log
    else:
        return Qs, g




if __name__ == "__main__":

    d = 3
    n = [10, 120, 8]
    r = 4
    dim = 2

    rng = np.random.default_rng(0)

    Xs = [rng.normal(size=(nk, dim)) for nk in n]

    # Probability measures
    a_list = []

    for nk in n:
        a = rng.random(nk)
        a /= np.sum(a)
        a_list.append(a)

    reg = 0.01
    alpha = 1e-6
    eta = 0.1
    gloabalC = pairwise_multi_marginal_cost(Xs)
    C_max = get_C_scale_exact(Xs)
    Qs, g, log = impcp_mirror_descent_pairwise(
        Xs=Xs,
        a_list=a_list,
        reg=reg,
        r=r,
        alpha=alpha,
        eta=eta,
        dykstra_tol=1e-8,
        dykstra_max_iter=500,
        max_iter=1000,
        tol=1e-7,
        C_scale=C_max,
        verbose=True,
        log=True,
    )

    print("Final cost:", compute_objective_pairwise(Xs, Qs, g, C_max))

    print("\nFinal g:")
    print(g)
    print("sum(g) =", np.sum(g))

    for k, Q in enumerate(Qs):
        print(f"\nQ^{k + 1} shape:", Q.shape)
        print(
            f"||Q^{k + 1} 1 - a^{k + 1}||_inf =",
            np.max(
                np.abs(
                    np.sum(Q, axis=1) - a_list[k]
                )
            ),
        )
        print(
            f"||Q^{k + 1}^T 1 - g||_inf =",
            np.max(
                np.abs(
                    np.sum(Q, axis=0) - g
                )
            ),
        )
