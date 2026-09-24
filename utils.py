import numpy as np
import ot

def pairwise_multi_marginal_cost(Xs):
    # returns the full tensor
    d = len(Xs)

    n = [X.shape[0] for X in Xs]

    C = np.zeros(n)

    for k in range(d):
        for l in range(k + 1, d):
            pairwise_cost = np.sum((Xs[k][:, None, :] - Xs[l][None, :, :]) ** 2,axis=-1)

            shape = [1] * d
            shape[k] = n[k]
            shape[l] = n[l]

            C += pairwise_cost.reshape(shape)

    return C
    
def marginal(T, i):
    # Compute the i-th marginal of tensor T
    nx = ot.backend.get_backend(T)
    d = len(T.shape)
    axes = tuple(j for j in range(d) if j != i)

    if len(axes) == 0:
        return T

    return nx.sum(T, axis=axes)

def marginal_error(T, list_a):
    errs = []
    for i, a in enumerate(list_a):
        errs.append(np.max(np.abs(marginal(T,i) - a)))
    return np.max(errs)

def normalize_costs(C_list):

    scale = max(np.max(C) for C in C_list)

    if scale <= 0:
        scale = 1.0

    C_normalized = [C / scale for C in C_list]

    return C_normalized

   

def get_C_scale(Xs, block_size=1024):
    #  Compute C_scale >= max(C)
    
    d = len(Xs)

    Xnorm2 = [np.sum(X * X, axis=1) for X in Xs]

    C_scale = 0.0

    for k in range(d):
        for l in range(k + 1, d):

            Xk = Xs[k]
            Xl = Xs[l]

            pair_max = -np.inf
            for start in range(0, Xk.shape[0], block_size):
                stop = min(start + block_size, Xk.shape[0])
                Xk_block = Xk[start:stop]
                D2 = (Xnorm2[k][start:stop, None] + Xnorm2[l][None, :] - 2.0 * (Xk_block @ Xl.T))
                pair_max = max(pair_max, np.max(D2))

            C_scale += pair_max

    return C_scale

def get_C_scale_exact(Xs, block_size=1024):
    # get max(C), may fail when C is too big
    d = len(Xs)
    sizes = [X.shape[0] for X in Xs]

    Xnorm2 = [np.sum(X * X, axis=1) for X in Xs]

    n_total = int(np.prod(sizes))
    C_scale = -np.inf

    for start in range(0, n_total, block_size):
        stop = min(start + block_size, n_total)

        indices = np.unravel_index(np.arange(start, stop), tuple(sizes))

        C_block = np.zeros(stop - start,dtype=float)

        for k in range(d):
            nk = Xnorm2[k][indices[k]]

            for l in range(k + 1, d):
                nl = Xnorm2[l][indices[l]]
                dot = np.sum(Xs[k][indices[k]] * Xs[l][indices[l]], axis=1)
                C_block += nk + nl - 2.0 * dot

        C_scale = max(C_scale,np.max(C_block))

    return C_scale

def compute_objective(C, T):
    nx = ot.backend.get_backend(C,T)
    return nx.sum(C * T)

def compute_objective_lr(C, Qs, g, C_scale, eps=1e-300):

    d = len(Qs)
    r = len(g)

    safe_g = np.maximum(g, eps)

    # p^(k) = Q^(k) / g
    Ps = [Q / safe_g[None, :] for Q in Qs]

    f = 0.0

    for s in range(r):
        tensor = C
        remaining_axes = list(range(d))
        for k in range(d):
            axis = remaining_axes.index(k)
            tensor = np.tensordot(tensor,Ps[k][:, s],axes=([axis], [0]))
            remaining_axes.pop(axis)
        f += g[s] * tensor

    return float(f) / C_scale

def compute_objective_pairwise(Xs, Qs, g, C_scale, eps=1e-20):

    d = len(Xs)

    safe_g = np.maximum(g, eps)
    Ps = [Q / safe_g[None, :] for Q in Qs]

    Xnorm2 = [np.sum(X * X, axis=1) for X in Xs]

    means = [X.T @ P for X, P in zip(Xs, Ps)]

    norm_means = [Xnorm2[k] @ Ps[k] for k in range(d)]

    total_norm = sum(norm_means)
    total_mean = sum(means)

    # c_s = sum_{k<l} E \|x_k-x_l\|^2
    cross_sum = np.sum(total_mean * total_mean, axis=0)

    for k in range(d):
        cross_sum -= np.sum(means[k] * means[k], axis=0)

    c = ((d - 1) * total_norm - cross_sum)

    return np.sum(g * c) / C_scale
    
def entropy_h(Qs,g,eps=1e-300):
    # H(x) = - sum_i x_i (log x_i - 1).
    value = 0.0

    for Q in Qs:
        X = np.maximum(Q, eps)
        value += np.sum(X * (np.log(X) - 1.0))

    G = np.maximum(g, eps)
    value += np.sum(G * (np.log(G) - 1.0))

    return -float(value)


def entropy_h2(Qs,eps=1e-300):
    # H(x) = - sum_i x_i (log x_i - 1).
    value = 0.0

    for Q in Qs:
        X = np.maximum(Q, eps)
        value += np.sum(X * (np.log(X) - 1.0))

    return -float(value)


def cp_tensor_from_Qs(Qs, g):
    d = len(Qs)
    r = len(g)

    shape = tuple(Q.shape[0] for Q in Qs)
    T = np.zeros(shape)

    for s in range(r):
        component = Qs[0][:, s]

        for k in range(1, d):
            component = np.multiply.outer(component, Qs[k][:, s])

        T += component / g[s]**(d - 1)

    return T
