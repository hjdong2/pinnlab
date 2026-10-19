
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_soln(x, t, u_pred, u_ref, path, slices=(0.25, 0.5, 0.75)):
    """save figs, top: reference, PINN, error over(t, x), bottom: u vs x at times in slices."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    panels = [
        (u_ref, "reference", "RdBu_r", -1, 1),
        (u_pred, "PINN", "RdBu_r", -1, 1),
        (np.abs(u_pred - u_ref), "|error|", "magma", None, None)
    ]
    for j, (field, title, cmap, vmin, vmax) in enumerate(panels):
        ax = axes[0, j]
        im = ax.imshow(field.T, origin="lower", aspect="auto",
                       extent=[t[0], t[-1], x[0], x[-1]], cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel="t", ylabel="x")

    for j, ts in enumerate(slices):
        ax = axes[1, j]
        k = int(np.argmin(np.abs(t - ts)))
        ax.plot(x, u_ref[k], "k-", lw=2, label="reference")
        ax.plot(x, u_pred[k], "r--", lw=2, label="PINN")
        ax.set(title=f"t = {t[k]:.2f}", xlabel="x", ylabel="u", ylim=(-1.1, 1.1))
        if j == 0:
            ax.legend()

    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_his(history, keys, path, extra=None):
    """plot PINN train history"""
    if extra is  None:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    else:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

    iters =[h["iter"] for h in history]

    ax.semilogy(iters, [h["loss"] for h in history], lw=2, label="total")
    for name in keys:
        ax.semilogy(iters, [h[name] for h in history], label=name)

    switch = next((h["iter"] for h in history if h["stage"] == "lbfgs"), None)
    if switch is not None:
        ax.axvline(switch, color='gray', linestyle=":", label="Adam --> L-BFGS")
    ax.legend()
    ax.set(title="loss", xlabel="iterations", ylabel="loss")

    if extra is not None:
        for i, (name, true_val) in enumerate(extra.items()):
            color = f"C{i}"
            ax2.plot(iters, [h[name] for h in history], color=color, label=f"{name} (learned)")
            ax2.axhline(true_val, color=color, linestyle="--", label=f"{name} (true)")
        ax2.set(xlabel="iterations", yscale="log")
        ax2.legend()

    fig.savefig(path, dpi=150)
    plt.close(fig)
