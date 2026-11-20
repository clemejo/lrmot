import numpy as np
import ot
import time
from utils import *
from ecpmot import expcp
from sinkhorn_mmot import sk_mmot
import matplotlib.pyplot as plt

## SETUP 
rng = np.random.default_rng(0)

n = [30, 30, 30, 30]
r = 7
dim = 2
reg = 0.005
global_max_iter = 1000
global_tol = 1e-6
Xs = []

for nk in n:
    X = rng.normal(size=(nk, dim))
    Xs.append(X)

SK = True
Bary = True
compute_T = True
print_marginals = False
verbose = False
save_datasets = False

if SK:
    C = pairwise_multi_marginal_cost(Xs)
    C = C / C.max()

C_scale = get_C_scale_exact(Xs)

a_list = []

for nk in n:
    a = rng.random(nk)
    a /= np.sum(a)
    a_list.append(a)

res_bary = []
res_bary_rdm = []
times_bary = []
times_bary_rdm = []
marg_bary = []
marg_bary_rdm = []

if SK:
    t1 = time.time()
    T_sk, _ = sk_mmot(C, a_list, reg, max_iter=15000, tol=1e-12) 
    t2 = time.time()

for seed in range(50):
    ## K means init
    t1 = time.time()
    Qs, g, _ = expcp(Xs, a_list, r, reg, max_outer_iter=global_max_iter, tol=global_tol, seed=seed, verbose=verbose)
    t2 = time.time()
    if compute_T:
        T_bary = cp_tensor_from_Qs(Qs, g)
    times_bary.append(t2-t1)
    marg_bary.append(marginal_error(T_bary, a_list))
    res_bary.append(compute_objective_pairwise(Xs, Qs, g, C_scale))

    ## Random init
    t1 = time.time()
    Qs, g, _ = expcp(Xs, a_list, r, reg, max_outer_iter=global_max_iter, tol=global_tol, seed=seed, verbose=verbose, init="random")
    if compute_T:
        T_bary_rdm = cp_tensor_from_Qs(Qs, g)
    t2 = time.time()
    if compute_T:
        T_bary_rdm = cp_tensor_from_Qs(Qs, g)
    times_bary_rdm.append(t2-t1)
    marg_bary_rdm.append(marginal_error(T_bary_rdm, a_list))
    res_bary_rdm.append(compute_objective_pairwise(Xs, Qs, g, C_scale))
    


    if print_marginals and compute_T:
        print("="*20+"\nMarginal errors\n"+"="*20)
        if SK:
            print("Sinkhorn:", marginal_error(T_sk, a_list))
        if Bary:
            print("Expl:", marginal_error(T_bary, a_list))
            print("Expl RDM:", marginal_error(T_bary_rdm, a_list))

val_sk = compute_objective(C,T_sk)
print("="*20)
if SK:
    print(f"SK: {val_sk:.4f}")
print("="*20)
if Bary:
    print("ECPLRMMOT:    ", " ".join(f"{value:5.4f}" for value in res_bary))
    print("ECPLRMMOT RDM:    ", " ".join(f"{value:5.4f}" for value in res_bary_rdm))
    print("Time mean kmeans", np.mean(times_bary))
    print("Time mean RDM", np.mean(times_bary_rdm))
    print("Err. marg mean kmeans", np.mean(marg_bary))
    print("Err. marg mean RDM", np.mean(marg_bary_rdm))
    print("val mean kmeans", np.mean(res_bary))
    print("val mean RDM", np.mean(res_bary_rdm))

data1 = np.array(res_bary)
data2 = np.array(res_bary_rdm)
mean1, std1 = np.mean(data1), np.std(data1, ddof=1)
mean2, std2 = np.mean(data2), np.std(data2, ddof=1)

x = np.linspace(min(data1.min(), data2.min()),max(data1.max(), data2.max()),500)

plt.figure(figsize=(9, 6))

plt.hist(data1, bins=20, density=True, alpha=0.3,color="royalblue", label="Distribution kmeans")
plt.hist(data2, bins=20, density=True, alpha=0.3,color="tomato", label="Distribution rdm")


plt.axvline(mean1, color="royalblue", linestyle="--",label=f"mean_kmeans = {mean1:.2f} ± {std1:.2f}")
plt.axvline(mean2, color="tomato", linestyle="--",label=f"mean_rdm = {mean2:.2f} ± {std2:.2f}")

plt.xlabel("Transport cost")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

if save_datasets:
    mean1 = np.mean(data1)
    std1 = np.std(data1, ddof=1)

    mean2 = np.mean(data2)
    std2 = np.std(data2, ddof=1)

    xmin = min(data1.min(), data2.min())
    xmax = max(data1.max(), data2.max())

    x = np.linspace(xmin, xmax, 500)

    nbins = 20

    bins = np.linspace(xmin, xmax, nbins + 1)

    hist1, edges = np.histogram(data1,bins=bins,density=True)
    hist2, _ = np.histogram(data2,bins=bins,density=True)

    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    np.savetxt("hist_distribution1.dat",np.column_stack((centers, hist1)),header="x density")
    np.savetxt("hist_distribution2.dat",np.column_stack((centers, hist2)),header="x density")
    print(  
        f"\\def\\meanone{{{mean1:.6f}}}\n"
        f"\\def\\stdone{{{std1:.6f}}}\n"
        f"\\def\\meantwo{{{mean2:.6f}}}\n"
        f"\\def\\stdtwo{{{std2:.6f}}}\n"
        f"\\def\\histwidth{{{width:.6f}}}"
    )



