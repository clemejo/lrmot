import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import *
import ot


def kmeanspp(X, r, seed=42):
    # kmeans++
    rng = np.random.default_rng(seed)

    n = X.shape[0]

    centers = np.empty((r, X.shape[1]))

    # first center at random
    idx = rng.integers(n)
    centers[0] = X[idx]

    # distance to closest center
    dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

    for j in range(1, r):
        total = dist_sq.sum()

        if total <= 0:
            idx = rng.integers(n)
        else:
            probs = dist_sq / total
            idx = rng.choice(n, p=probs)

        centers[j] = X[idx]

        new_dist_sq = np.sum((X - centers[j]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)

    return centers


def d_kmeans_z(Xs, r, seed=42):

    d = len(Xs)

    # k-means++ for each marginal
    tilde_Xs = [kmeanspp(X, r, seed=seed) for X in Xs]

    # first point cloud as reference
    Z = tilde_Xs[0].copy()

    for k in range(1, d):
        # cost matrix between the reference centers and centers of the kth point cloud
        C = np.sum((tilde_Xs[0][:, None, :] - tilde_Xs[k][None, :, :]) ** 2, axis=2)

        _, col_ind = linear_sum_assignment(C)
        matched = tilde_Xs[k][col_ind]

        Z += matched

    Z /= d

    return Z

def sk_bary_g(r,C_list,a_list,reg,iter_max=500,tol=1e-5,warmstart=None,log=False):
    # Takes normalized cost matrices
    nx = ot.backend.get_backend(C_list[0])
    d = len(a_list)
    if log:
        log = {"err": []}

    K_list = [np.exp(-C / reg) for C in C_list] 
    if warmstart == None:
        u_list = [np.ones(C.shape[0]) for C in C_list]
        v_list = [np.ones(r) for _ in range(d)]
    else:
        u_list = warmstart[0]
        v_list = warmstart[1]

    if log:
        log["K"] = [K_list]
        log["u"] = [u_list]
        log["v"] = [v_list]

    g = np.ones(r) / r # Initialization
    tiny = np.finfo(float).tiny

    err = 42
    cpt = 0
    while (err > tol) and (cpt < iter_max):
        cpt += 1
        list_uprev = u_list
        list_vprev = v_list

        Kv_list = [nx.dot(K,v) for K,v in zip(K_list,v_list)]

        u_list = [a / nx.maximum(Kv, tiny) for a, Kv in zip(a_list,Kv_list)]
 
        KTu_list = [nx.dot(K.T,u) for K,u in zip(K_list,u_list)]

        log_g = nx.sum(nx.stack([nx.log(nx.maximum(KTu, tiny)) for KTu in KTu_list]), axis=0) / d
        g = np.maximum(nx.exp(log_g), 1e-12) 
        g = g / nx.sum(g)  # sinkhorn scaling invariance

        v_list = [g / nx.maximum(KTu, tiny) for KTu in KTu_list]

        if (any(nx.any(KTu == 0) for KTu in KTu_list) or any(nx.any(nx.isnan(u)) for u in u_list) or any(nx.any(nx.isnan(v)) for v in v_list)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            u_list = list_uprev
            v_list = list_vprev
            break

        if cpt % 20 == 0:
            errors = [] 
            for K, u, v, a in zip(K_list, u_list, v_list, a_list):
                row_marginal = u * nx.dot(K, v) 
                row_error = nx.norm(row_marginal - a) 
                errors.append(row_error) # Maximum marginal violation over all distributions 
                err = max(errors)
            if log:
                log["err"].append(err)

        if log:
            log["niter"] = cpt
            log["u"].append(u_list)
            log["v"].append(v_list)

    Qs = [u.reshape((-1, 1)) * K * v.reshape((1, -1)) for u,K,v in zip(u_list,K_list,v_list)]
    if log:
        return Qs, g, log
    else:
        return Qs, g

def expcp(Xs, weights, r, reg,max_outer_iter=100,max_sinkhorn_iter=1300,tol=1e-7,sinkhorn_tol=1e-7,C_scale=None,init=None,seed=42,verbose=False,log=False):

    rng = np.random.default_rng(seed)  # eventual random initialization

    d = len(Xs)

    p = Xs[0].shape[1] # points in R^p

    if log:
        log = {"cpt": 0}

    if init == "random":
        Z = rng.random((r,p))
    else:
        Z = d_kmeans_z(Xs, r, seed=seed)
    if C_scale is None:
        C_scale = get_C_scale(Xs) # not exact but faster

    def compute_costs_z(Z):
        C_raw = [np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=2) for X in Xs]
        return [C / C_scale for C in C_raw]
    
    C_list = compute_costs_z(Z)
    prev_loss = 42
    warmstart_skg = None
    for outer_it in range(max_outer_iter):
        # BarySinkhorn
        Qs, g, log_skg = sk_bary_g(r,C_list,weights,reg=reg/d,iter_max=max_sinkhorn_iter,tol=sinkhorn_tol,warmstart=warmstart_skg,log=True)
        warmstart_skg = [log_skg["u"][-1],log_skg["v"][-1]]

        # Update z
        numerator = np.zeros_like(Z)
        for X, Q in zip(Xs, Qs):
            numerator += Q.T @ X

        denominator = d * g[:, None]
        Z = numerator / np.maximum(denominator, np.finfo(float).tiny)

        # Update costs
        C_list = compute_costs_z(Z)

        # convergence
        transport_normalized = 0.0
        for X, Q in zip(Xs, Qs):
            C_raw = np.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=2)
            C = C_raw / C_scale
            transport_normalized += np.sum(Q * C)
        curr_loss = d*transport_normalized - reg*entropy_h2(Qs)
        err = abs(curr_loss - prev_loss) / abs(prev_loss) if prev_loss != 0.0 else np.nan

        if verbose:
            transport_normalized = 0.0
            k = 0
            for X, Q in zip(Xs, Qs):
                transport_normalized += np.sum(Q * C_list[k])
                k += 1
            objective = d*transport_normalized - reg*entropy_h2(Qs)
            print(f"Outer iteration {outer_it:4d}: d*transport = {d*transport_normalized:.10e}, entropy = {- reg*entropy_h2(Qs):.10e}, objective = {objective:.10e}, relative evolution = {err:.3e}")
        if log:
            log["cpt"] = outer_it
        if err < tol:
            break

        prev_loss = curr_loss

    if log:
        return Qs, g, Z, log
    else:
        return Qs, g, Z


if __name__ == "__main__":
    seed = 1
    d = 3
    n = [10, 120, 8]
    r = 4
    dim = 2

    rng = np.random.default_rng(seed)

    Xs = []

    for nk in n:
        X = rng.normal(size=(nk, dim))
        Xs.append(X)

    gloabalC = pairwise_multi_marginal_cost(Xs)
    C_max = get_C_scale_exact(Xs)

    weights = []
    
    for nk in n:
        a = rng.random(nk)
        a /= np.sum(a)
        weights.append(a)

    reg = 0.01

    Qs, g, Z = expcp(Xs=Xs,weights=weights,r=r,reg=reg,max_outer_iter=100,max_sinkhorn_iter=1000,tol=1e-6,sinkhorn_tol=1e-8,seed=1,verbose=True)
    print("Final cost:", compute_objective_pairwise(Xs, Qs, g, C_max))
    print("\nBarycenter weights:")
    print(g)

    print("\nBarycenter support:")
    print(Z)

    for k, Q in enumerate(Qs):
        print(f"\nQ^{k + 1} shape:", Q.shape)
        print(f"||Q^{k + 1} 1 - a^{k + 1}||_inf =", np.max(np.abs(np.sum(Q, axis=1) - weights[k])))
        print(f"||Q^{k + 1}^T 1 - g||_inf =", np.max(np.abs(np.sum(Q, axis=0) - g)))
