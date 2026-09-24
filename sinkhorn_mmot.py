import numpy as np
import ot
import time
from utils import *
from scipy.special import logsumexp

def sk_mmot(C,a_list,reg,log=False,max_iter=1000,tol=10**-6,max_time=180,warmstart=None):
    t_start = time.time()
    nx = ot.backend.get_backend(C)

    d = len(a_list)

    if warmstart == None:
        f_list = [nx.zeros(a.shape[0]) for a in a_list]
    else:
        f_list = warmstart

    log_K = -C / reg

    if log:
        log = {"err": []}
        log["K"] = [log_K]
        log["f"] = [f_list]
        log["elapsed_time"] = []

    def compute_log_T(f_list):
        log_T = log_K.copy()

        for i in range(d):
            shape = [1] * d
            shape[i] = len(f_list[i])

            log_T += f_list[i].reshape(shape)

        return log_T

    def log_marginal(log_T, i):
        axes = tuple(j for j in range(d) if j != i)
        return logsumexp(log_T, axis=axes)

    log_T = compute_log_T(f_list)
    err = 42
    cpt = 0
    total_time = 0
    while (err > tol) and (cpt <= max_iter) and (total_time < max_time):   
        for i in range(d):
            log_marg = log_marginal(log_T, i)
            f_list[i] += (nx.log(a_list[i]) - log_marg)

            # Update T 
            log_T = compute_log_T(f_list)

        errs = []

        for i in range(d):
            log_marg = log_marginal(log_T, i)
            marg = nx.exp(log_marg)

            err_i = nx.sum(nx.abs(marg - a_list[i]))
            errs.append(err_i)

        err = max(errs)
    
        cpt += 1
        total_time = time.time() - t_start

        if log:
            log["err"].append(err)
            log["elapsed_time"].append(total_time)
            log["f"].append(f_list.copy())

    
    duals = [reg * x.copy() for x in f_list]

    for k in range(d):
        duals[k] -= nx.mean(duals[k])
    if log:
        log["cpt"] = cpt
        return nx.exp(log_T), duals, log
    
    return nx.exp(log_T), duals

def greedy_sk_mmot(C,a_list,reg,log=False,max_iter=1000,tol=1e-6,max_time=180,warmstart=None):
    t_start = time.time()
    nx = ot.backend.get_backend(C)

    d = len(a_list)
    log_K = -C / reg
    log_a = [nx.log(a) for a in a_list]

    if warmstart is None:
        f_list = [nx.zeros(a.shape) for a in a_list]
    else:
        f_list = [nx.copy(f) for f in warmstart]

    if log:
        log = {
            "err": [],
            "K": [],
            "f": [f_list.copy()],
            "elapsed_time": [],
        }

    def compute_log_T(f_list):
        log_T = log_K.copy()

        for k in range(d):
            shape = [1] * d
            shape[k] = len(f_list[k])
            log_T += f_list[k].reshape(shape)

        return log_T

    def log_marginal(log_T, k):
        axes = tuple(j for j in range(d) if j != k)
        return nx.logsumexp(log_T, axis=axes)

    log_T = compute_log_T(f_list)

    err = float("inf")
    cpt = 0
    total_time = 0.0
    K = None
    while (err > tol and cpt < max_iter and total_time < max_time):
        if K is None:
            # One computation of all the marginals of the full tensor 
            log_margs = [log_marginal(log_T, k) for k in range(d)]
            scores = [nx.sum(a_list[k] * (log_a[k] - log_margs[k])) for k in range(d)]
            K = max(range(d), key=lambda k: float(scores[k]))
            log_marg_K = log_margs[K]
        else:
            log_A = nx.logsumexp(log_T, axis=K)
            remaining = [k for k in range(d) if k != K]

            reduced_index = {k: j for j, k in enumerate(remaining)}
            log_margs = {}

            for k in remaining:
                j = reduced_index[k]
                axes = tuple(q for q in range(d - 1) if q != j)

                log_margs[k] = nx.logsumexp(log_A,axis=axes)

            scores = {k: nx.sum(a_list[k] * (log_a[k] - log_margs[k])) for k in remaining}

            K = max(scores, key=lambda k: float(scores[k]))
            log_marg_K = log_margs[K]

        delta = log_a[K] - log_marg_K
        f_list[K] += delta
        shape = [1] * d
        shape[K] = len(delta)
        log_T += delta.reshape(shape)

        if cpt % 10 == 0:
            errs = []
            for k in range(d):
                if k == K:
                    continue

                log_marg = log_marginal(log_T, k)
                marg = nx.exp(log_marg)

                errs.append(float(nx.sum(nx.abs(marg - a_list[k]))))

            err = max(errs, default=0.0)

        cpt += 1
        total_time = time.time() - t_start

        if log:
            log["err"].append(err)
            log["K"].append(K)
            log["elapsed_time"].append(total_time)
            log["f"].append(f_list.copy())

    duals = [reg * x.copy() for x in f_list]
    mean_dual = nx.mean(nx.concatenate([x.reshape(-1) for x in duals]))
    duals = [x - mean_dual for x in duals]

    T = nx.exp(log_T)

    if log:
        log["cpt"] = cpt
        return T, duals, log

    return T, duals

if __name__ == "__main__":

    d = 4
    n = [25, 50, 25, 50]
    dim = 2

    rng = np.random.default_rng(0)

    Xs = []

    for nk in n:
        X = rng.normal(size=(nk, dim))
        Xs.append(X)

    gloabalC = pairwise_multi_marginal_cost(Xs)
    gloabalC = gloabalC / gloabalC.max()

    weights = []
    
    for nk in n:
        a = rng.random(nk)
        a /= np.sum(a)
        weights.append(a)

    # Entropic regularization
    epsilon = 0.01

    T, _, log_sk = sk_mmot(gloabalC, weights, reg=epsilon, log=True) 
    print("Final cost SK:", compute_objective(gloabalC, T))

    for k, a in enumerate(weights):
        print(f"||Marg_k(T) - a^{k + 1}||_inf =", np.max(np.abs(marginal(T,k) - a)))

    T_gre, _, log_gre = greedy_sk_mmot(gloabalC, weights, reg=epsilon, log=True) 
    print("Final cost Greedy SK:", compute_objective(gloabalC, T_gre))

    for k, a in enumerate(weights):
        print(f"||Marg_k(T) - a^{k + 1}||_inf =", np.max(np.abs(marginal(T_gre,k) - a)))

        
    print(
        f"SK: {log_sk['cpt']:4d} it, {log_sk['elapsed_time'][-1]:7.2f}s | "
        f"Greedy SK: {log_gre['cpt']:4d} it, {log_gre['elapsed_time'][-1]:7.2f}s | "
    )

        