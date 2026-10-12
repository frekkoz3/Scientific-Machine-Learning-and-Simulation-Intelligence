import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

M = sp.load_npz("M.npz").toarray()
K = sp.load_npz("K.npz").toarray()
C = sp.load_npz("C.npz").toarray()

S10 = np.load("snapshots_Pe10.npz")["S"]
S1000 = np.load("snapshots_Pe1000.npz")["S"]
x = np.load("snapshots_Pe10.npz")["x"]

energy_target = 0.9999


def pod(S):
    lam, W = la.eigh(S.T @ M @ S)
    lam, W = lam[::-1], W[:, ::-1]
    p = lam > 1e-14 * lam[0]
    lam, W = lam[p], W[:, p]
    e = np.cumsum(lam / lam.sum())
    N = np.searchsorted(e, energy_target) + 1
    return S @ W[:, :N] / np.sqrt(lam[:N]), N, lam, e


Z10, N10, lam10, e10 = pod(S10)
Z1000, N1000, lam1000, e1000 = pod(S1000)

print(f"Pe = 10: N = {N10}, retained energy = {100*e10[N10-1]:.4f}%")
print(f"Pe = 1000: N = {N1000}, retained energy = {100*e1000[N1000-1]:.4f}%")


# POD eigenvalues and cumulative energy
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].semilogy(lam10 / lam10[0], label="Pe = 10")
axs[0].semilogy(lam1000 / lam1000[0], label="Pe = 1000")
axs[0].set(
    xlabel="POD mode",
    ylabel=r"$\lambda_i/\lambda_1$",
    title="POD eigenvalue decay"
)
axs[0].legend()

axs[1].plot(100 * e10, label="Pe = 10")
axs[1].plot(100 * e1000, label="Pe = 1000")
axs[1].axhline(
    100 * energy_target, ls="--",
    label=f"{100*energy_target:.2f}% target"
)
axs[1].plot(N10, 100 * e10[N10-1], "o")
axs[1].plot(N1000, 100 * e1000[N1000-1], "o")
axs[1].set(
    xlabel="Number of POD modes",
    ylabel="Cumulative energy [%]",
    title="Cumulative POD energy",
    ylim=(0, 100.5)
)
axs[1].legend()

plt.tight_layout()
plt.show()


# ROM integration
dt, T = 1e-6, 0.5
n_steps = round(T / dt)


def rom(Z, S, Pe):
    a = Z.T @ M @ S[:, 0]
    K_N = Z.T @ K @ Z

    start = time.perf_counter()

    for _ in range(n_steps):
        u = Z @ a
        a -= dt * (K_N @ a / Pe + Z.T @ (C @ (u * u)))

    return Z @ a, time.perf_counter() - start


u10, t10 = rom(Z10, S10, 10)
u1000, t1000 = rom(Z1000, S1000, 1000)

F10, F1000 = S10[:, -1], S1000[:, -1]

err10 = np.sqrt((F10-u10) @ M @ (F10-u10))
err1000 = np.sqrt((F1000-u1000) @ M @ (F1000-u1000))

print(f"ROM runtime: Pe = 10: {t10:.3f} s, Pe = 1000: {t1000:.3f} s")
print(f"Pe = 10:   final-time error = {err10:.6e}")
print(f"Pe = 1000: final-time error = {err1000:.6e}")


# FOM vs ROM
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axs[0].plot(x, F10, label="FOM")
axs[0].plot(x, u10, "--", label="ROM")
axs[0].set_title(
    rf"Pe = 10, $\|u_{{FOM}}-u_{{ROM}}\|_M={err10:.2e}$"
)

axs[1].plot(x, F1000, label="FOM")
axs[1].plot(x, u1000, "--", label="ROM")
axs[1].set_title(
    rf"Pe = 1000, $\|u_{{FOM}}-u_{{ROM}}\|_M={err1000:.2e}$"
)

for ax in axs:
    ax.set_xlabel("x")

axs[0].set_ylabel("u")
axs[0].legend()

plt.tight_layout()
plt.show()


# POD modes
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for i in range(N10):
    axs[0].plot(x, Z10[:, i], label=rf"$\zeta_{i+1}$")

for i in range(N1000):
    axs[1].plot(x, Z1000[:, i], label=rf"$\zeta_{i+1}$")

axs[0].set(title="Pe = 10", xlabel="$x$", ylabel=r"$\zeta_i(x)$")
axs[1].set(title="Pe = 1000", xlabel="$x$")

plt.tight_layout()
plt.show()


# Full POD bases
def full_pod(S):
    lam, W = la.eigh(S.T @ M @ S)
    lam, W = lam[::-1], W[:, ::-1]
    p = lam > 1e-14 * lam[0]
    return S @ W[:, p] / np.sqrt(lam[p])


Z10f, Z1000f = full_pod(S10), full_pod(S1000)

Ns = [1, 10, 20, 30, 40, 50, 60, 70, 80]

err10s, err1000s, p10s, p1000s = [], [], [], []

for N in Ns:
    V10, V1000 = Z10f[:, :N], Z1000f[:, :N]

    P10 = V10 @ (V10.T @ M @ F10)
    P1000 = V1000 @ (V1000.T @ M @ F1000)

    p10s.append(np.sqrt((F10-P10) @ M @ (F10-P10)))
    p1000s.append(np.sqrt((F1000-P1000) @ M @ (F1000-P1000)))

    u10, _ = rom(V10, S10, 10)
    u1000, _ = rom(V1000, S1000, 1000)

    err10s.append(np.sqrt((F10-u10) @ M @ (F10-u10)))
    err1000s.append(np.sqrt((F1000-u1000) @ M @ (F1000-u1000)))


# Convergence
plt.figure(figsize=(6, 4))

plt.semilogy(Ns, err10s, "o-", label="ROM, Pe = 10")
plt.semilogy(Ns, p10s, "o--", label="Projection, Pe = 10")
plt.semilogy(Ns, err1000s, "s-", label="ROM, Pe = 1000")
plt.semilogy(Ns, p1000s, "s--", label="Projection, Pe = 1000")

plt.xlabel("Number of POD modes")
plt.ylabel("Error at final time")
plt.legend()
plt.tight_layout()
plt.show()
