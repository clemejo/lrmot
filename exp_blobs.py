import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tuckmmot import tucker_mmot

def get_blobs(n, ranks, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    Xs = []
    for ni, ri in zip(n, ranks):
        sizes = np.full(ri, ni // ri, dtype=int)
        sizes[:ni % ri] += 1
        centers = rng.normal(loc=0.0, scale=3.0, size=(ri, dim))
        X = np.vstack([centers[j] + rng.normal(scale=0.3, size=(sizes[j], dim)) for j in range(ri)])
        rng.shuffle(X, axis=0)
        Xs.append(X)
    return Xs

## SETUP 
rng = np.random.default_rng(0)

n = [30,50,60]
ranks = [4,5,6]
global_max_iter = 5000
global_tol = 1e-6
global_max_time = 300
Xs = get_blobs(n, ranks)
a_list = []
for nk in n:
    a = np.ones(nk) / nk
    a_list.append(a)

result = tucker_mmot(Xs,a_list,ranks,reg=0.05,gamma=1,max_iter=global_max_iter,tol=global_tol,max_time=global_max_time,greedy=False,verbose=True)
G = result["G"]
Qs = result["Q"]

Zs = []
for k in range(len(n)):
    gk = marginal(G, k)          
    Qk = Qs[k]                   
    Xk = Xs[k]                   
    # z_s^(k) = 1/g_s^(k) \sum_i Q_is^(k) x_i^(k)
    Zk = (Qk.T @ Xk) / gk[:, None]
    Zs.append(Zk)

fig, ax = plt.subplots(figsize=(7, 7))

colors = plt.cm.tab10(np.arange(len(Xs)))

for k, (Xk, Zk) in enumerate(zip(Xs, Zs)):
    ax.scatter(Xk[:, 0],Xk[:, 1],color=colors[k],alpha=0.25,s=40,label=fr"$X^{{({k+1})}}$")
    ax.scatter(Zk[:, 0],Zk[:, 1],color=colors[k],marker="*",s=180,edgecolor="black",linewidth=0.7,label=fr"$\mu^{{({k+1})}}$")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# for k, (Xk, Zk) in enumerate(zip(Xs, Zs), start=1):
#     np.savetxt(f"X_{k}.txt",Xk,fmt="%.8f",header="x y")
#     np.savetxt(f"Z_{k}.txt",Zk,fmt="%.8f",header="x y")