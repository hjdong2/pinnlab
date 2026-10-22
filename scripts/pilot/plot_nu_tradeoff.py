
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = json.loads(Path("scratch/nu_sweep.json").read_text())
nus = np.array([r["nu"] for r in rows])
train_err = np.array([r["l1_vis"] for r in rows])
tot_err = np.array([r["l1_inv"] for r in rows])

p, logC = np.polyfit(np.log(nus), np.log(train_err), 1)
C = np.exp(logC)
nu_star = (C / (4 * np.log(2)))**(1 / (1 - p))
print(f"p {p} C {C}")

nu_line = np.logspace(-4.5, -1.5, 100)

fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
ax.loglog(nu_line, 4 * np.log(2) * nu_line, "k-", label="viscosity error 4 ln 2 ν (exact)")
ax.loglog(nus, train_err, "o", label="training error (L1 vs viscous)")
ax.loglog(nus, tot_err, "s", label="total error (L1 vs inviscid)")
# ax.axvline(nu_star, color="gray", linestyle=":", label=f"crossover ν {nu_star:.1e}")
ax.set(xlabel="ν (artificial viscosity)", ylabel="L1 error at t = 0.5", title="Shock: viscosity vs training error")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.savefig("scratch/nu_tradeoff.png", dpi=150)
plt.close(fig)
