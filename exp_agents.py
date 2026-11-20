import numpy as np
from utils import *
import matplotlib.pyplot as plt

def project_C1_partial(Qs, g, marginals, alpha, eps=1e-100):

    Qs_proj = [np.maximum(Q, eps).copy() for Q in Qs]

    for k, a in marginals.items():
        Q = Qs_proj[k]
        row_sums = Q.sum(axis=1)

        Qs_proj[k] *= (a / np.maximum(row_sums, eps))[:, None]

    g_proj = np.maximum(g, alpha)

    return Qs_proj, g_proj

def project_C2(Qs, g, eps=1e-300):

    d = len(Qs)

    col_sums = [np.maximum(Q.sum(axis=0), eps) for Q in Qs]

    log_g_star = (np.log(np.maximum(g, eps)) + sum(np.log(cs) for cs in col_sums)) / (d + 1)

    g_star = np.exp(log_g_star)
    Qs_proj = [Q * (g_star / cs)[None, :] for Q, cs in zip(Qs, col_sums)]

    return Qs_proj, g_star

def make_dykstra_projector_partial(marginals,alpha=1e-12,tol=1e-8,max_iter=1000,eps=1e-300,log_clip=700):
    def safe_exp(log_x):
        return np.exp(np.clip(log_x, -log_clip, log_clip))

    def projector(Qs_init, g_init):
        log_Qs = [np.log(np.maximum(Q, eps)) for Q in Qs_init]
        log_g = np.log(np.maximum(g_init, eps))

        log_p1_Q = [np.zeros_like(Q) for Q in Qs_init]
        log_p1_g = np.zeros_like(g_init)

        log_p2_Q = [np.zeros_like(Q) for Q in Qs_init]
        log_p2_g = np.zeros_like(g_init)

        for it in range(max_iter):

            old_log_Qs = [x.copy() for x in log_Qs]
            old_log_g = log_g.copy()

            log_Y_Q = [x + p for x, p in zip(log_Qs, log_p1_Q)]
            log_Y_g = log_g + log_p1_g

            Y_Q = [safe_exp(x) for x in log_Y_Q]
            Y_g = safe_exp(log_Y_g)

            Z_Q, Z_g = project_C1_partial(Y_Q, Y_g, marginals, alpha, eps)

            log_Z_Q = [np.log(np.maximum(x, eps)) for x in Z_Q]
            log_Z_g = np.log(np.maximum(Z_g, eps))

            log_p1_Q = [y - z for y, z in zip(log_Y_Q, log_Z_Q)]
            log_p1_g = log_Y_g - log_Z_g

            log_Y2_Q = [z + p for z, p in zip(log_Z_Q, log_p2_Q)]
            log_Y2_g = log_Z_g + log_p2_g

            Y2_Q = [safe_exp(x) for x in log_Y2_Q]
            Y2_g = safe_exp(log_Y2_g)

            X_Q, X_g = project_C2(Y2_Q, Y2_g, eps)

            log_X_Q = [np.log(np.maximum(x, eps)) for x in X_Q]
            log_X_g = np.log(np.maximum(X_g, eps))

            log_p2_Q = [y - x for y, x in zip(log_Y2_Q, log_X_Q)]
            log_p2_g = log_Y2_g - log_X_g

            log_Qs = log_X_Q
            log_g = log_X_g

            numerator = max(max(np.max(np.abs(x - y)) for x, y in zip(log_Qs, old_log_Qs)),np.max(np.abs(log_g - old_log_g)))

            if numerator < tol:
                break

        Qs_out = [safe_exp(x) for x in log_Qs]
        g_out = safe_exp(log_g)

        return Qs_out, g_out

    return projector

def compute_gradient_tree(Qs, g, edges):
    ps = [Q / g[None, :] for Q in Qs]

    grad_Q = [np.zeros_like(Q) for Q in Qs]
    grad_g = np.zeros_like(g)

    for (k, l), C in edges.items():
        grad_Q[k] += C @ ps[l]
        grad_Q[l] += C.T @ ps[k]

        grad_g -= np.sum(ps[k] * (C @ ps[l]), axis=0)

    return grad_Q, grad_g

def initialize_tree(marginals, dims, r, seed=42):

    rng = np.random.default_rng(seed)

    mass = np.sum(next(iter(marginals.values())))

    g = rng.random(r) + 1.0
    g = mass * g / g.sum()

    Qs = []

    for k, nk in enumerate(dims):

        Q = rng.random((nk, r)) + 1e-3

        if k in marginals:
            a = marginals[k]
            Q = Q * (a / Q.sum(axis=1))[:, None]
        else:
            Q = Q * (g / Q.sum(axis=0))[None, :]

        Qs.append(Q)

    return Qs, g

def compute_objective_tree(Qs, g, edges, eps=1e-300):
    Ps = [Q / np.maximum(g, eps)[None, :] for Q in Qs]
    c = np.zeros(len(g))

    for (k, l), C_kl in edges.items():
        c += np.sum(Ps[k] * (C_kl @ Ps[l]),axis=0)

    return np.dot(g, c)

def impcp_mirror_descent_tree(
    edges,
    marginals,
    dims,
    reg,
    r,
    alpha=1e-8,
    eta=0.1,
    dykstra_tol=1e-8,
    dykstra_max_iter=1000,
    max_iter=1000,
    tol=1e-6,
    seed=42,
    verbose=False,
    log=False
):

    Qs, g = initialize_tree(marginals, dims, r, seed)
    projector = make_dykstra_projector_partial(marginals=marginals,alpha=alpha,tol=dykstra_tol,max_iter=dykstra_max_iter)

    Qs, g = projector(Qs, g)

    history = {"objective": [],"regularized_objective": [],"change": []}

    prev_loss = None

    for t in range(max_iter):
        grad_Qs, grad_g = compute_gradient_tree(Qs, g, edges)

        # Mirror descent step
        exponent = 1.0 - eta * reg
        K_Qs = []
        for Q, grad in zip(Qs, grad_Qs):
            log_K = (exponent * np.log(np.maximum(Q, 1e-300)) - eta * grad)
            K_Qs.append(np.exp(np.clip(log_K, -700, 700)))

        log_Kg = (exponent * np.log(np.maximum(g, 1e-300)) - eta * grad_g)
        K_g = np.exp(np.clip(log_Kg, -700, 700))

        # Dykstra projection
        Qs_new, g_new = projector(K_Qs, K_g)

        # Convergence
        change = max(max(np.max(np.abs(Qnew - Qold)) for Qnew, Qold in zip(Qs_new, Qs)),np.max(np.abs(g_new - g)))

        Qs, g = Qs_new, g_new

        f_value = compute_objective_tree(Qs, g, edges) 

        objective = f_value - reg * entropy_h(Qs, g)

        if log:
            history["objective"].append(f_value)
            history["regularized_objective"].append(objective)
            history["change"].append(change)

        if verbose and (t % 10 == 0 or t == max_iter - 1):
            print(
                f"iter={t:5d} | "
                f"f={f_value:.8e} | "
                f"objective={objective:.8e} | "
                f"change={change:.3e}")

        if prev_loss is not None:
            err = abs(objective - prev_loss) / max(abs(prev_loss), 1e-15)
            if err < tol and change < tol:
                if verbose:
                    print(f"Mirror Descent converged at iteration {t}.")
                break

        prev_loss = objective

    if log:
        return Qs, g, history

    return Qs, g

def generate_measurements(X, B_list, seed=1234):
    rng = np.random.default_rng(seed)
    tau, N = X.shape
    S = len(B_list)
    Phi = []
    for t in range(tau):
        Phi_t = []
        for s in range(S):
            B = B_list[s]
            probabilities = B[X[t], :]
            observations = np.array([rng.choice(2, p=prob) for prob in probabilities])
            histogram = np.bincount(observations,minlength=2)
            Phi_t.append(histogram.astype(float))
        Phi.append(Phi_t)
    return Phi


def build_grid(size=10):
    positions = np.array([[i, j] for i in range(1, size + 1) for j in range(1, size + 1)], dtype=float)
    n = size * size
    adjacency = np.zeros((n, n))

    def idx(i, j):
        return i * size + j

    for i in range(size):
        for j in range(size):
            k = idx(i, j)

            if i > 0:
                adjacency[k, idx(i-1, j)] = 1

            if i < size - 1:
                adjacency[k, idx(i+1, j)] = 1

            if j > 0:
                adjacency[k, idx(i, j-1)] = 1

            if j < size - 1:
                adjacency[k, idx(i, j+1)] = 1

    return positions, adjacency


def build_random_walk(adjacency, lazy=True):
    degrees = adjacency.sum(axis=1)
    P = adjacency / degrees[:, None]
    if lazy:
        P = 0.5 * np.eye(len(P)) + 0.5 * P
    return P

def compute_schrodinger_bridge(P,tau,start=0,end=99,eps=1e-300):
    n = P.shape[0]

    if tau < 2:
        raise ValueError("tau must be at least 2.")

    if not (0 <= start < n and 0 <= end < n):
        raise ValueError("start and end must be valid state indices.")

    h_list = [None] * tau

    h_list[-1] = np.zeros(n)
    h_list[-1][end] = 1.0

    for t in range(tau - 2, -1, -1):
        h_list[t] = P @ h_list[t + 1]

    if h_list[0][start] <= eps:
        raise ValueError(
            "No positive-probability path connects start to end "
            "in the available number of time steps."
        )

    mu_list = [None] * tau
    M_list = []
    A_list = []

    mu_list[0] = np.zeros(n)
    mu_list[0][start] = 1.0

    for t in range(tau - 1):
        h_next = h_list[t + 1]

        A = np.zeros_like(P, dtype=float)

        numerator = P * h_next[None, :]

        # Normalize rows with positive probability mass
        row_sums = numerator.sum(axis=1)
        valid_rows = row_sums > eps

        A[valid_rows, :] = numerator[valid_rows, :] / row_sums[valid_rows, None]

        # Transport matrix
        M = mu_list[t][:, None] * A

        # Next marginal
        mu_next = M.sum(axis=0)

        total_mass = mu_next.sum()

        if total_mass <= eps:
            raise ValueError(f"Numerical failure: zero total mass at time {t + 1}.")

        mu_next /= total_mass

        A_list.append(A)
        M_list.append(M)
        mu_list[t + 1] = mu_next

    return mu_list, M_list, A_list

def generate_sensor_positions(S=15,low=1.0,high=10.0,seed=123):
    rng = np.random.default_rng(seed)
    return rng.uniform(low,high,size=(S, 2))


def build_observation_matrices(node_positions,sensor_positions,sigma=1):
    B_list = []
    for sensor in sensor_positions:
        distances = np.linalg.norm(node_positions - sensor[None, :],axis=1)

        p_detect = np.minimum(0.99,2.0 * np.exp(-distances/sigma))

        # column 0: detected
        # column 1: not detected
        B = np.column_stack([p_detect,1.0 - p_detect])
        B_list.append(B)

    return B_list

def simulate_agents(A_list, N, start=0, seed=42):
    rng = np.random.default_rng(seed)
    tau = len(A_list) + 1
    X = np.zeros((tau, N), dtype=int)
    X[0, :] = start

    for t, A in enumerate(A_list):
        for a in range(N):
            probs = A[X[t, a], :].copy()
            probs /= probs.sum()
            X[t + 1, a] = rng.choice(n, p=probs)

    return X

# PARAMETERS

N = 100
grid_size = 10
tau = grid_size*3

n = grid_size**2

S = 15
r = 100
reg = 1
max_iter = 500
sigma = grid_size/10
start = 0
end = n - 1

times = [5, 10, 20, 28]



node_positions, adjacency = build_grid(size=grid_size)

assert adjacency.shape == (n, n)
assert node_positions.shape == (n, 2)

P = build_random_walk(adjacency, lazy=True)

mu_prob, M_prob, A_bridge = compute_schrodinger_bridge(P,tau=tau,start=start,end=end)

# Convert probability distributions into agent counts
mu_list = [N * mu for mu in mu_prob]
M_list = [N * M for M in M_prob]

# used in article
sensor_positions = np.array([[2.1,1],[8.2,1.8],[4.5,2.2],[5.8,2.9],[1.1,2.9],[2.7,4.8],[4.8,4.9],[4.1,6],[9.2,6],[1.9,7],[7.5,7.2],[3.6,8.8],[6.4,9.2],[9.2,9.4],[7.4,5]])

#sensor_positions = generate_sensor_positions(S=S,low=1.0,high=float(grid_size),seed=123)

B_list = build_observation_matrices(node_positions,sensor_positions,sigma=sigma)

X = simulate_agents(A_bridge,N=N,start=start,seed=42)

Phi = generate_measurements(X,B_list,seed=1234)

# Temporal costs: random-walk prior
C_time = -np.log(np.maximum(P, 1e-300))

# Observation costs
C_obs = [-np.log(np.maximum(B, 1e-300)) for B in B_list]

assert np.isclose(mu_list[0][start], N)
assert np.isclose(mu_list[-1][end], N)

# Mass conservation
for t in range(tau):
    assert np.isclose(mu_list[t].sum(), N)

# Transport consistency
for t in range(tau - 1):
    assert np.allclose(M_list[t].sum(axis=1), mu_list[t])
    assert np.allclose(M_list[t].sum(axis=0), mu_list[t + 1])

# Observation mass
for t in range(tau):
    for s in range(S):
        assert np.isclose(Phi[t][s].sum(), N)

# Simulated endpoints
assert np.all(X[0] == start)
assert np.all(X[-1] == end)

print("All checks passed!")

# Dimensions of all marginals:
# tau spatial variables of dimension n
# tau*S observation variables of dimension 2

dims = [n] * tau + [2] * (tau * S)

edges = {}

# Temporal edges
for t in range(tau - 1):
    edges[(t, t + 1)] = C_time.copy()

# Sensor edges
for t in range(tau):
    for s in range(S):
        k = t
        l = tau + t * S + s

        edges[(k, l)] = C_obs[s].copy()


# PRESCRIBED MARGINALS
marginals = {}

# Initial distribution
mu_initial = np.zeros(n)
mu_initial[start] = N
marginals[0] = mu_initial

# Final distribution
mu_final = np.zeros(n)
mu_final[end] = N
marginals[tau - 1] = mu_final

# Sensor observations
for t in range(tau):
    for s in range(S):
        k = tau + t * S + s
        marginals[k] = Phi[t][s]



Qs, g = impcp_mirror_descent_tree(
    edges=edges,
    marginals=marginals,
    dims=dims,
    reg=reg,
    r=r,
    eta=0.03,
    max_iter=max_iter,
    verbose=True
)

# Recover estimated spatial distributions
mu_est = [Qs[t].sum(axis=1) for t in range(tau)]


## Plotting

true_grids = [np.asarray(mu_list[t - 1]).reshape(grid_size, grid_size) for t in times]

estimated_grids = [np.asarray(mu_est[t - 1]).reshape(grid_size, grid_size) for t in times]

vmin = min(min(x.min() for x in true_grids),min(x.min() for x in estimated_grids))
vmax = max(max(x.max() for x in true_grids),max(x.max() for x in estimated_grids))


def print_distribution_summary(mu, name):
    grid = np.asarray(mu).reshape(grid_size, grid_size)

    x = np.arange(1, grid_size+1)
    y = np.arange(1, grid_size+1)

    total = grid.sum()

    mean_x = np.sum(grid * x[None, :]) / total
    mean_y = np.sum(grid * y[:, None]) / total

    max_idx = np.unravel_index(np.argmax(grid), grid.shape)

    print(f"{name}:")
    print(f"  Center of mass: ({mean_x:.2f}, {mean_y:.2f})")
    print(f"  Maximum location: ({max_idx[1]+1}, {max_idx[0]+1})")
    print(f"  Maximum value: {grid.max():.3f}")

for t in times:
    print(f"\nTime {t}")
    print_distribution_summary(mu_list[t-1], "True")
    print_distribution_summary(mu_est[t-1], "Estimated")




data = []

for j, t in enumerate(times):
    true_grid = np.asarray(true_grids[j])
    estimated_grid = np.asarray(estimated_grids[j])

    n_y, n_x = true_grid.shape

    for iy in range(n_y):
        for ix in range(n_x):
            data.append([
                t,
                ix + 1,
                iy + 1,
                true_grid[iy, ix],
                estimated_grid[iy, ix]
            ])

data = np.asarray(data)

header = "time x y true_distribution estimated_distribution"

np.savetxt(
    "agent_distributions.txt",
    data,
    fmt=["%d", "%d", "%d", "%.8f", "%.8f"],
    delimiter="\t",
    header=header,
    comments=""
)

fig, axes = plt.subplots(2,4,figsize=(16, 8),constrained_layout=True)

max_value = max(np.max(grid) for grid in list(true_grids) + list(estimated_grids))
size_scale = 8.0

x, y = np.meshgrid(np.arange(1, grid_size+1),np.arange(1, grid_size+1))

for j, t in enumerate(times):

    true_grid = np.asarray(true_grids[j])
    estimated_grid = np.asarray(estimated_grids[j])

    # True distribution
    axes[0, j].scatter(
        x.ravel(),
        y.ravel(),
        s=size_scale * true_grid.ravel(),
        color="blue",
        alpha=0.85,
        edgecolors="none"
    )

    axes[0, j].set_title(f"True ensemble: t = {t}")

    # Estimated distribution
    axes[1, j].scatter(
        x.ravel(),
        y.ravel(),
        s=size_scale * estimated_grid.ravel(),
        color="blue",
        alpha=0.85,
        edgecolors="none"
    )

    axes[1, j].set_title(f"Estimated ensemble: t = {t}")

    # Common formatting
    for ax in [axes[0, j], axes[1, j]]:
        ax.set_xlim(0.5, grid_size+0.5)
        ax.set_ylim(0.5, grid_size+0.5)

        ax.set_xticks(range(1, grid_size+1))
        ax.set_yticks(range(1, grid_size+1))

        ax.set_aspect("equal")

        # Blue grid
        ax.grid(True,color="lightblue",linestyle="-",linewidth=0.8,alpha=0.8)
        ax.set_axisbelow(True)

fig.suptitle("True and estimated agent distributions",fontsize=16)

plt.show()
