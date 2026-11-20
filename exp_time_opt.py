import numpy as np
import ot
import time
from utils import *
from ecpmot import expcp
from cpmot import impcp_mirror_descent_pairwise
from sinkhorn_mmot import sk_mmot, greedy_sk_mmot
from tuckmmot import tucker_mmot, evaluate_tucker, marginal_error_tucker

def mean_std(values):
    values = np.asarray(values, dtype=float)
    return np.mean(values), np.std(values)

def generate_gaussian_mixture_samples(n = [10,20,30,40], d=2, seed=None):
    rng = np.random.default_rng(seed)

    mu1 = np.array([
        [0.0, 0.0] + (d-2)*[0],
        [0.0, 1.0]+ (d-2)*[0],
        [1.0, 1.0]+ (d-2)*[0]
    ])

    mu2 = np.array([
        [0.5, 0.5]+ (d-2)*[0],
        [-0.5, 0.5]+ (d-2)*[0]
    ])

    mu3 = np.array([
        [0.5, 0.5]+ (d-2)*[0],
        [-0.5, 0.5]+ (d-2)*[0],
        [0, 0.5]+ (d-2)*[0],
        [-0.5,0]+ (d-2)*[0]
    ])

    mu4 = np.array([
        [0.5, -0.5]+ (d-2)*[0],
        [-0.5, -0.5]+ (d-2)*[0]
    ])

    Sigma = 0.05 * np.eye(d)

    ix1 = rng.choice(3, size=n[0], p=[1/3, 1/3, 1/3])
    ix2 = rng.choice(2, size=n[1], p=[1/2, 1/2])
    ix3 = rng.choice(4, size=n[2], p=[1/4, 1/4, 1/4, 1/4])
    ix4 = rng.choice(2, size=n[3], p=[1/2, 1/2])

    X1 = mu1[ix1] + rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n[0])
    X2 = mu2[ix2] + rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n[1])
    X3 = mu3[ix3] + rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n[2])
    X4 = mu4[ix4] + rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n[3])

    return [X1, X2, X3, X4]

## SETUP 
rng = np.random.default_rng(0)

n = [25, 50, 25, 50]
list_r = [2]
ranks = [len(n)*[r] for r in list_r]
dim = 2
reg = 0.005
global_max_iter = 3000
global_tol = 1e-6
global_max_time = 60    # Tucker can take time, and is run twice per rank per iteration
nb_iter = 3  # 30 in the paper 

SK = False
GSK = False
ExpCP = True
MD = True
Tucker = True
print_marginals = True
verbose = False

results = {"SK": {}, "GSK": {}, "ExpCP": {}, "MD": {}, "Tucker": {}, "Tuckergsk": {}}
for i in range(nb_iter):

    Xs = generate_gaussian_mixture_samples(n, dim, seed=i)

    if SK or GSK:
        C = pairwise_multi_marginal_cost(Xs)
        C = C / C.max()

    C_scale = get_C_scale_exact(Xs)
    a_list = []

    for nk in n:
        a = rng.random(nk)
        a /= np.sum(a)
        a_list.append(a)

    res_ExpCP = []
    res_md = []
    res_stro = []
    res_tuck = []
    if SK:
        t1 = time.time()
        T_sk, _, log = sk_mmot(C, a_list, reg, max_iter=global_max_iter, tol=global_tol, max_time=global_max_time, log=True) 
        t2 = time.time()
        time_sk = t2 - t1
        value_sk = compute_objective(C, T_sk)
        marginal_sk = marginal_error(T_sk, a_list)
        niter_sk = log["cpt"]
        print("Time SK", t2-t1)
        results["SK"].setdefault(None, []).append((value_sk, marginal_sk, niter_sk, time_sk))

    if GSK:
        t1 = time.time()
        T_gsk, _, log = greedy_sk_mmot(C, a_list, reg, max_iter=global_max_iter, tol=global_tol, max_time=global_max_time, log=True) 
        t2 = time.time()
        time_gsk = t2 - t1
        value_gsk = compute_objective(C, T_gsk)
        marginal_gsk = marginal_error(T_gsk, a_list)
        niter_gsk = log["cpt"]
        print("Time GSK", t2-t1)
        results["GSK"].setdefault(None, []).append((value_gsk, marginal_gsk, niter_gsk, time_gsk))

    for i, r in enumerate(list_r):
        if ExpCP:
            t1 = time.time()
            Qs, g, _, log = expcp(Xs, a_list, r, reg, max_outer_iter=global_max_iter, tol=global_tol, seed=1, verbose=verbose, log=True)
            t2 = time.time()
            time_ExpCP = t2 - t1
            cpt_ExpCP = log["cpt"]
            T_ExpCP = cp_tensor_from_Qs(Qs, g)
            marginal_ExpCP = marginal_error(T_ExpCP, a_list)
            value_ExpCP = compute_objective_pairwise(Xs, Qs, g, C_scale)
            results["ExpCP"].setdefault(r, []).append((value_ExpCP, marginal_ExpCP, cpt_ExpCP, time_ExpCP))

        if MD:
            t1 = time.time()
            Qs, g, log = impcp_mirror_descent_pairwise(Xs, a_list, reg, r, max_iter=global_max_iter, tol=global_tol, verbose=verbose, seed=1, log=True)
            t2 = time.time()
            T_md = cp_tensor_from_Qs(Qs, g)
            cpt_MD = log["cpt"]
            time_MD = t2 - t1
            value_MD = compute_objective_pairwise(Xs, Qs, g, C_scale)
            marginal_MD = marginal_error(T_md, a_list)
            results["MD"].setdefault(r, []).append((value_MD, marginal_MD, cpt_MD, time_MD))

        if Tucker:
            t1 = time.time()
            result = tucker_mmot(Xs, a_list, ranks[i], reg=0.04, gamma=0.1,max_iter=global_max_iter,tol=global_tol,max_time=global_max_time, greedy=False,verbose=False)
            t2 = time.time()
            time_Tucker = t2 - t1
            cpt_Tucker = result["cpt"]
            G = result["G"]
            Qs = result["Q"]
            value_Tucker = evaluate_tucker(Xs, Qs, G, C_scale, eps=1e-300)
            marginal_Tucker = marginal_error_tucker(result, a_list)
            results["Tucker"].setdefault(r, []).append((value_Tucker, marginal_Tucker, cpt_Tucker, time_Tucker))

            t1 = time.time()
            result = tucker_mmot(Xs, a_list, ranks[i], reg=0.04, gamma=0.1,max_iter=global_max_iter,tol=global_tol,max_time=global_max_time, greedy=True,verbose=False)
            t2 = time.time()
            time_Tucker = t2 - t1
            cpt_Tucker = result["cpt"]
            G = result["G"]
            Qs = result["Q"]
            value_Tucker = evaluate_tucker(Xs, Qs, G, C_scale, eps=1e-300)
            marginal_Tucker = marginal_error_tucker(result, a_list)
            results["Tuckergsk"].setdefault(r, []).append((value_Tucker, marginal_Tucker, cpt_Tucker, time_Tucker))

def print_row(method, values):
    values = np.asarray(values, dtype=float)

    mean_value, std_value = mean_std(values[:, 0])
    mean_marg, _ = mean_std(values[:, 1])
    mean_iter, std_iter = mean_std(values[:, 2])
    mean_time, std_time = mean_std(values[:, 3])

    value_str = f"${mean_value:.3f} \\pm {std_value:.3f}$"

    if mean_marg > 0 and np.isfinite(mean_marg):
        exponent = int(np.floor(np.log10(abs(mean_marg))))
        marginal_str = f"$10^{{{exponent}}}$"
    else:
        marginal_str = "$-$"

    iter_str = f"${mean_iter:.0f} \\pm {std_iter:.0f}$"

    time_str = f"${mean_time:.1f} \\pm {std_time:.1f}$"

    print(f"{method} & {value_str} & {marginal_str} & {iter_str} & {time_str} \\\\")

if SK:
    print_row("Sinkhorn", results["SK"][None])

if GSK:
    print_row("Greedy Sinkhorn", results["GSK"][None])

if ExpCP:
    for r in list_r:
        print_row("ExpCP", results["ExpCP"][r])

if MD:
    for r in list_r:
        print_row("ImpCP", results["MD"][r])
if Tucker:
    for r in list_r:
        print_row("Tucker GSK", results["Tuckergsk"][r])

if Tucker:
    for r in list_r:
        print_row("Tucker SK", results["Tucker"][r])

