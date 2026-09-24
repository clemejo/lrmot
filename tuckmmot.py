import numpy as np
import time
from sinkhorn_mmot import sk_mmot, greedy_sk_mmot
from utils import pairwise_multi_marginal_cost, get_C_scale_exact, entropy_h2 as entropy_h

## Implementation for pairwise costs, we do not use pseudo inverse of P^{(k)}

def normalize_columns(Q, eps=1e-15):
    g = Q.sum(axis=0)
    P = Q / np.maximum(g[None, :], eps)
    return P, g


def kl_div(p, q, eps=1e-15):
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return np.sum(p * (np.log(p) - np.log(q)) - p + q)


def normalize_cost(C, eps=1e-12):
    C = np.asarray(C, dtype=float)

    shift = np.min(C)
    scale = np.max(C) - shift

    scale = max(float(scale), eps)
    C_norm = (C - shift) / scale

    return C_norm, scale

def evaluate_tucker(Xs, Qs, G, C_scale, eps=1e-300):
    # for pairwise costs
    d = len(Xs)
    gs = []
    for k in range(d):
        axes = tuple(j for j in range(d) if j != k)
        gs.append(np.maximum(np.sum(G, axis=axes), eps))

    Ps = [Q / g[None, :] for Q, g in zip(Qs, gs)]
    means = [X.T @ P for X, P in zip(Xs, Ps)]

    second = [np.sum(X * X, axis=1) @ P for X, P in zip(Xs, Ps)]

    norm_expectations = [np.dot(g, q) for g, q in zip(gs, second)]

    total_cost = 0.0

    for k in range(d):
        for l in range(k + 1, d):
            pair_cost = norm_expectations[k] + norm_expectations[l]
            axes = tuple(j for j in range(d) if j not in (k, l))
            G_kl = np.sum(G, axis=axes)
            cross = np.sum(G_kl * (means[k].T @ means[l]))
            pair_cost -= 2.0 * cross
            total_cost += pair_cost

    return total_cost / C_scale

def marginal_error_tucker(result, a_list):
    Qs = result["Q"]
    err = [np.max(np.abs(Q.sum(axis=1) - a_list[k])) for k, Q in enumerate(Qs)]
    return np.max(err)

def initialize_random_Q(a,g,rng,max_iter=10000,tol=1e-12,eps=1e-300):
    a = np.asarray(a, dtype=float)
    g = np.asarray(g, dtype=float)

    n = len(a)
    r = len(g)

    logK = rng.normal(size=(n, r))
    logK -= np.max(logK)

    K = np.exp(logK)

    u = np.ones(n)
    v = np.ones(r)

    for _ in range(max_iter):
        u = a / np.maximum(K @ v, eps)
        v = g / np.maximum(K.T @ u, eps)

        Q = u[:, None] * K * v[None, :]

        row_err = np.max(np.abs(Q.sum(axis=1) - a))
        col_err = np.max(np.abs(Q.sum(axis=0) - g))

        if max(row_err, col_err) < tol:
            break

    Q = u[:, None] * K * v[None, :]
    P = Q / np.maximum(g[None, :], eps)

    return Q, P


def factor_cost_pairwise(k, Xs, P_list, G):
    """
    Effective cost C_{Q^{(k)}} for updating Q^(k) 
    Ck[i, s] = E[C(X_1,...,X_d) | X_k = Xs[k][i], S_k = s]
    for pairwise costs
    """
    d = len(Xs)
    rk = P_list[k].shape[1]

    mu = []
    second = []
    for l in range(d):
        X = Xs[l]
        P = P_list[l]
        # E[X_l | S_l=t]
        mu_l = P.T @ X
        # E[\|X_l\|^2 | S_l=t]
        second_l = P.T @ np.sum(X * X, axis=1)
        mu.append(mu_l)
        second.append(second_l)

    axes_sum = tuple(j for j in range(d) if j != k)
    gk = G.sum(axis=axes_sum)

    Ck = np.zeros((Xs[k].shape[0], rk))

    cond_mu = {}
    cond_second = {}
    for l in range(d):
        if l == k:
            continue

        axes_sum = tuple(j for j in range(d) if j not in (k, l))
        Mkl = G.sum(axis=axes_sum)
        # Reorder to (k,l)
        if k > l:
            Mkl = Mkl.T
        # Mkl[s,t] = P(S_k=s,S_l=t)
        W = Mkl / np.maximum(gk[:, None], 1e-300)
        cond_mu[l] = W @ mu[l]
        cond_second[l] = W @ second[l]

    Xk = Xs[k]
    xk_sq = np.sum(Xk * Xk, axis=1)
    for l in range(d):
        if l == k:
            continue

        Ck += (xk_sq[:, None] + cond_second[l][None, :] - 2.0 * Xk @ cond_mu[l].T)

    others = [l for l in range(d) if l != k]

    for ii, l in enumerate(others):
        for m in others[ii + 1:]:
            axes_sum = tuple(j for j in range(d) if j not in (k, l, m))
            Mklm = G.sum(axis=axes_sum)
            retained = [j for j in range(d) if j not in axes_sum]
            # Reorder to (k,l,m).
            perm = [retained.index(k),retained.index(l),retained.index(m)]
            Mklm = np.transpose(Mklm, perm)
            Wklm = Mklm / np.maximum(gk[:, None, None],1e-300)
            ml = cond_second[l]
            mm = cond_second[m]
            # E[mu_l(S_l)^T mu_m(S_m) | S_k=s]
            cross = np.einsum("sij,id,jd->s",Wklm,mu[l],mu[m])
            pair_cost = (ml + mm - 2.0 * cross)
            Ck += pair_cost[None, :]

    return Ck


def update_P(k, Xs, P_list, Q_list, G, alpha, a, gamma=1.0, tau=1.0,cost_scale=1.0, eps=1e-300,n_proj_iter=20):
    Q_old = Q_list[k]

    Ck = factor_cost_pairwise(k, Xs, P_list, G)
    Ck_norm = Ck / max(cost_scale, eps)

    g_old = Q_old.sum(axis=0)
    g_old = np.maximum(g_old, eps)
    g_old /= g_old.sum()

    Cbar = np.sum(Q_old * Ck_norm, axis=0) / g_old
    grad_input = Ck_norm - Cbar[None, :]
    grad = grad_input + alpha[None, :]

    logK = np.log(np.maximum(Q_old, eps)) - gamma * grad
    K = np.exp(np.clip(logK, -700.0, 700.0))

    # Semi-relaxed KL projection
    v = np.ones(Q_old.shape[1])

    exponent = tau / (tau + 1.0 / gamma)

    for _ in range(n_proj_iter):
        u = a / np.maximum(K @ v, eps)
        v = (g_old / np.maximum(K.T @ u, eps)) ** exponent

    Q_new = u[:, None] * K * v[None, :]

    g_new = Q_new.sum(axis=0)
    g_new = np.maximum(g_new, eps)
    g_new /= g_new.sum()

    P_new = Q_new / np.maximum(g_new[None, :], eps)

    return Q_new, P_new, g_new, Ck, grad

def latent_cost_pairwise(Xs, P_list):
    """
    Construct C_G
    for pairwise costs
    """
    d = len(Xs)

    latent_means = []
    latent_variances = []

    for k in range(d):

        P = P_list[k]
        X = Xs[k]

        mu = P.T @ X
        second = P.T @ np.sum(X * X, axis=1)

        variance = second - np.sum(mu * mu, axis=1)
        variance = np.maximum(variance, 0.0)

        latent_means.append(mu)
        latent_variances.append(variance)

    shape = tuple(P.shape[1] for P in P_list)

    C = np.zeros(shape)

    grids = np.meshgrid(*[np.arange(P.shape[1]) for P in P_list],indexing="ij")

    for k in range(d):
        for ell in range(k + 1, d):
            Xk = latent_means[k][grids[k]]
            Xl = latent_means[ell][grids[ell]]
            C += np.sum((Xk - Xl) ** 2,axis=-1)

    # Each conditional variance appears in d-1 pairs
    for k in range(d):
        C += (d - 1) * latent_variances[k][grids[k]]

    return C


def input_marginal_from_Q(Q):
    return Q.sum(axis=1)


def core_marginal(G, k):
    # Mode-k marginal of the latent core
    d = G.ndim
    axes = tuple(j for j in range(d) if j != k)
    return G.sum(axis=axes)


def print_factor_err(Q,a,g,k):
    q_row_error = np.max(np.abs(Q.sum(axis=1) - a))
    q_col_error = np.max(np.abs(Q.sum(axis=0) - g))
    print(f"k={k}: Q row error = {q_row_error:.3e}, Q col error = {q_col_error:.3e}")


def print_core_err(G, g_list):
    print("\nMarginal errors of the core:")
    for k, g in enumerate(g_list):
        marginal = core_marginal(G, k)
        error = np.max(np.abs(marginal - g))
        print(f"k={k}: {error:.3e}")


def tucker_mmot(Xs,a_list,ranks,reg=0.05,gamma=0.5,tau=0.2,max_iter=500,core_max_iter=10000,tol=1e-8,core_tol=1e-8,max_time = 600,greedy=False,seed=0,verbose=True):

    rng = np.random.default_rng(seed)

    d = len(Xs)
    t_start = time.time()

    Q_list = []
    P_list = []
    g_list = []

    for k in range(d):
        rk = ranks[k]
        g = rng.uniform(0.8,1.2,rk)
        g /= g.sum()

        Q, P = initialize_random_Q(a_list[k],g,rng)

        Q_list.append(Q)
        P_list.append(P)
        g_list.append(g)

    G = g_list[0].copy()

    for k in range(1, d):
        G = np.multiply.outer(G,g_list[k])

    C_G = latent_cost_pairwise(Xs,P_list)

    C_G_norm, cost_scale = normalize_cost(C_G)
    if greedy:
        G, alpha = greedy_sk_mmot(C_G_norm,g_list,reg=reg,max_iter=core_max_iter,tol=core_tol)
    else:
        G, alpha = sk_mmot(C_G_norm,g_list,reg=reg,max_iter=core_max_iter,tol=core_tol)

    obj_history = []
    input_cost_history = []

    previous_objective = np.inf

    for it in range(max_iter):
        C_G = latent_cost_pairwise(Xs,P_list)

        candidates = []
        for k in range(d):
            Q_candidate,P_candidate,g_candidate,Ck,grad = update_P(k=k,Xs=Xs,P_list=P_list,Q_list=Q_list,G=G,alpha=alpha[k],a=a_list[k],gamma=gamma,tau=tau,cost_scale=cost_scale)
            candidates.append((Q_candidate,P_candidate,g_candidate,Ck,grad))

        Q_new_list = [candidates[k][0] for k in range(d)]
        P_new_list = [candidates[k][1] for k in range(d)]
        g_new_list = [candidates[k][2] for k in range(d)]

        C_G_new = latent_cost_pairwise(Xs,P_new_list)

        C_G_norm, cost_scale = normalize_cost(C_G)
        if greedy:
            G, alpha = greedy_sk_mmot(C_G_norm,g_list,reg=reg,max_iter=core_max_iter,tol=core_tol)
        else:
            G, alpha = sk_mmot(C_G_norm,g_list,reg=reg,max_iter=core_max_iter,tol=core_tol)
        Q_list = Q_new_list
        P_list = P_new_list
        g_list = g_new_list

        C_G = C_G_new

        input_cost = np.sum(C_G * G)
        input_cost_history.append(input_cost)

        max_factor_error = 0.0

        for k in range(d):
            Q = Q_list[k]
            P = P_list[k]
            g = g_list[k]
            a = a_list[k]

            row_error = np.max(np.abs(Q.sum(axis=1) - a))
            col_error = np.max(np.abs(Q.sum(axis=0) - g))
            p_col_error = np.max(np.abs(P.sum(axis=0) - 1.0))
            pg_error = np.max(np.abs(P @ g - a))
            max_factor_error = max(max_factor_error,row_error,col_error,p_col_error,pg_error)

        core_error = 0.0

        for k in range(d):
            marginal = core_marginal(G,k)
            error = np.max(np.abs(marginal - g_list[k]))
            core_error = max(core_error,error)

        C_G_norm = C_G / max(cost_scale,1e-300)

        normalized_cost = np.sum(C_G_norm * G)

        core_entropy = entropy_h(G)
        objective = (normalized_cost - reg * core_entropy)
        obj_history.append(objective)
        
        reg_input = - reg * cost_scale
        regularized_objective = (input_cost + reg_input * core_entropy)
 
        if verbose and (it % 100 == 0 or it == max_iter - 1):
            print(
                f"iter {it:4d} | "
                f"Obj = {regularized_objective:.10f} | "
                f"C = {input_cost:.10f} | "
                f"epsH = {reg_input * core_entropy:.10f}"
            )

        total_time = time.time() - t_start
        if it > 0:
            rel_change = (abs(objective - previous_objective) / max(1.0,abs(previous_objective)))

            if ((rel_change < tol and max_factor_error < 10 * core_tol and core_error < 10 * core_tol) or total_time > max_time):
                if verbose:
                    print(f"Converged at iteration {it}.")
                break

        previous_objective = objective

    return {"Q": Q_list,"P": P_list,"g": g_list,"G": G,"alpha": alpha,"obj_history": obj_history,"input_cost_history": input_cost_history,"cpt": it}

if __name__ == "__main__":

    rng = np.random.default_rng(42)

    n = [20, 10, 20]
    d = len(n)

    r = 5
    ranks = [r] * d

    dim = 2

    reg = 0.05
    Xs = []

    for nk in n:
        X = rng.normal(size=(nk, dim))
        Xs.append(X)

    a_list = []

    for nk in n:
        a = rng.random(nk)
        a /= np.sum(a)
        a_list.append(a)


    result = tucker_mmot(Xs,a_list,ranks,reg=0.03,gamma=0.1,tau=1,max_iter=3000,tol=1e-6,verbose=True)

    Q_list = result["Q"]
    P_list = result["P"]
    g_list = result["g"]
    G = result["G"]

    print("\nFinal latent marginals:")

    for k, g in enumerate(g_list):
        print(f"k={k}: sum={g.sum():.8f}, min={g.min():.3e}, max={g.max():.3e}")

    print("\nCore shape:", G.shape)
    print("Core mass:", G.sum())

    print_core_err(G,g_list)

    print("\nFactor feasibility:")

    for k in range(d):
        print_factor_err(Q_list[k],a_list[k],g_list[k],k=k)

    print("\ninput marginal errors:")

    for k in range(d):

        input_marginal = input_marginal_from_Q(Q_list[k])

        error = np.max(np.abs(input_marginal - a_list[k]))

        print(f"k={k}: {error:.3e}")
    gloabalC = pairwise_multi_marginal_cost(Xs)
    C_max = get_C_scale_exact(Xs)
    print(f"\nFinal input pairwise cost: {evaluate_tucker(Xs, Q_list, G, C_max, eps=1e-300):.10f}")
