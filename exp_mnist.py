import numpy as np
import ot
import time
from utils import *
from ecpmot import expcp
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def print_img(vec):
    img = np.asarray(vec).reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def print_imgs(vec_list, title="z as images"):
    n = len(vec_list)
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = np.atleast_1d(axes).flatten()
    axes[0].set_title(title)
    for i, vec in enumerate(vec_list):
        img = np.asarray(vec).reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

def get_transport(Qs, g, indices):
    d = len(Qs)
    return np.sum(np.prod([Qs[k][indices[k], :] for k in range(d)], axis=0) / g**(d - 1))

def get_matrix(Qs, g, k, l):
    return Qs[k] @ np.diag(1 / g) @ Qs[l].T


(X_train, y_train), (X_test, y_test) = mnist.load_data()

images = np.concatenate([X_train, X_test], axis=0)
labels = np.concatenate([y_train, y_test])

# Flatten 28x28 images -> 784-dimensional vectors
images = images.reshape(len(images), -1).astype(np.float64)

images /= 255.0

rng = np.random.default_rng(0)

d = 20
# 70000 images
# n = d * [int(len(images)/d)]
n = d * [1500]
r = 10
reg = 0.001
verbose = True

Xs = []
Ys = []

used = np.zeros(len(images), dtype=bool)

for nk in n:
    nk_label = nk // 10

    indices_k = []

    for digit in range(10):
        available = np.where((labels == digit) & (~used))[0]
        idx = rng.choice(available, size=nk_label, replace=False)
        indices_k.extend(idx)
        used[idx] = True

    indices_k = np.array(indices_k)
    #rng.shuffle(indices_k)

    Xs.append(images[indices_k])
    Ys.append(labels[indices_k])

# print how much of each number in each measure
# for k, Yk in enumerate(Ys):
#     counts = np.bincount(Yk, minlength=10)
#     print(f"X[{k}]:", counts)

a_list = []

for nk in n:
    a = np.ones(nk) / nk
    a_list.append(a)

# C is too big to use exact scale, but it is not problematic
C_scale = get_C_scale(Xs)

t1 = time.time()
Qs, g, z_exp = expcp(Xs, a_list, r, reg, max_outer_iter=1000, tol=1e-6, seed=1, verbose=verbose)
print("g =", g)
t2 = time.time()
print("Time ECP (d)", t2-t1)



C01 = ot.dist(Xs[0],Xs[1])
C01 = C01 / C01.max()

T01 = get_matrix(Qs, g, 0, 1)

# Best competitor for large scale in d=2 case
from Riem import solve_balanced_low_rank_ot
t1 = time.time()
result, manifold, log_rcg = solve_balanced_low_rank_ot(C=C01,a=a_list[0],b=a_list[1],rank=r,max_iterations=1000,tol=1e-6,min_gradient_norm=1e-6,beta_rule="HestenesStiefel",verbosity=1,seed=1,log=True)
Q_rcg, R_rcg = result.point
g_rcg = Q_rcg.sum(axis=0)
T01_lr = Q_rcg @ np.diag(1.0 / g) @ R_rcg.T
t2 = time.time()
print("Time RCG", t2 - t1)
z_riemcg = 0.5 * (Q_rcg.T @ Xs[0] + R_rcg.T @ Xs[1]) / g[:, None]


# ExpCP with 2 measures
t1 = time.time()
Qs_2, g_2, z_exp_2 = expcp([Xs[0], Xs[1]], [a_list[0], a_list[1]], r, reg, max_outer_iter=1000, tol=1e-6, seed=1, verbose=verbose)
T01_lr2 = get_matrix(Qs_2, g_2, 0, 1)
t2 = time.time()
print("Time ECP (2)", t2-t1)

print_imgs(z_exp, "ECP (d)")
print_imgs(z_riemcg, "RiemCG")
print_imgs(z_exp_2, "ECP (2)")

print("Total cost RCG a1 -> a2", compute_objective(C01, T01_lr))
print("Total cost ECP (2)", compute_objective(C01, T01_lr2))
print("Total cost ECP (d)", compute_objective(C01, T01))

fig, axes = plt.subplots(1, 3, figsize=(10, 6))

axes[0].imshow(T01)
axes[0].set_title("ECP (d)")
axes[1].imshow(T01_lr)
axes[1].set_title("RiemCG")
axes[2].imshow(T01_lr2)
axes[2].set_title("ECP (2)")


plt.tight_layout()
plt.show()
