import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time


def assemble(n_el):
    h, n = 1 / n_el, n_el - 1
    tri = lambda a, b, c: sp.diags(
        [a, b, c], [-1, 0, 1], shape=(n, n), format="csc"
    )
    M = h / 6 * tri(1, 4, 1)
    K = 1 / h * tri(-1, 2, -1)
    C = 0.5 * tri(-1, 0, 1)
    return M, K, C


if __name__ == "__main__":
    # Parameters
    n_el, T, dt = 1000, 0.5, 1e-6
    save_every, plot_every = 1000, 1000
    Pe_values = [10, 100, 1000]
    x0, w0 = 0.2, 0.01

    h = 1 / n_el
    x = np.linspace(0, 1, n_el + 1)[1:-1]
    M, K, C = assemble(n_el)
    M_solve = spla.splu(M).solve
    n_steps = round(T / dt)

    # Live visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    line, = ax.plot(x, np.zeros_like(x))
    ax.set(xlim=(0, 1), ylim=(-0.2, 1.1), xlabel="x", ylabel="u")

    # FOM simulations and snapshot storage
    for Pe in Pe_values:
        assert dt <= h**2 * Pe / 6, f"dt too large for Pe={Pe}"
        assert Pe * h / 2 <= 1, f"Mesh Peclet number too large for Pe={Pe}"

        A = K / Pe + C
        u = np.exp(-((x - x0) / w0)**2)
        S, times = [], []

        start = time.perf_counter()
        for k in range(n_steps + 1):
            if k % save_every == 0:
                S.append(u.copy())
                times.append(k * dt)

            if k % plot_every == 0:
                line.set_ydata(u)
                ax.set_title(f"Pe = {Pe}, t = {k * dt:.3f}")
                plt.pause(1e-3)

            u -= dt * M_solve(A @ u)

        S = np.array(S).T
        print(f"Pe = {Pe}: runtime = {time.perf_counter() - start:.3f} s")
        np.savez(f"snapshots_Pe{Pe}.npz", S=S, t=times, x=x, Pe=Pe)
        print(f"Saved snapshots for Pe={Pe}: {S.shape}")

    # Final overview: snapshots at selected times
    plt.ioff()
    fig2, axs = plt.subplots(
        1, len(Pe_values), figsize=(4 * len(Pe_values), 3),
        sharey=True, squeeze=False
    )

    for ax, Pe in zip(axs[0], Pe_values):
        data = np.load(f"snapshots_Pe{Pe}.npz")
        S, times, x = data["S"], data["t"], data["x"]

        indices = np.arange(S.shape[1])[::max(1, S.shape[1] // 5)]
        for j in indices:
            ax.plot(x, S[:, j], label=f"t = {times[j]:.2f}")

        ax.set(title=f"Pe = {Pe}", xlabel="x")

    axs[0, 0].set_ylabel("u")
    axs[0, 0].legend()
    fig2.tight_layout()
    plt.show()
