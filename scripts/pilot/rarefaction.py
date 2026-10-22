
import json
import time
from pathlib import Path

import numpy as np
import torch

from pinnlab.models import MLP
from pinnlab.problems import burgers, riemann
from pinnlab.train import train

torch.set_num_threads(2)
ul, ur = 0.0, 1.0
s = (ul + ur) / 2
x = np.linspace(-1, 1, 8001)

weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
results = []

for nu in [1.0e-2, 1.0e-3]:
    torch.manual_seed(0)
    model = MLP(2, 1, 20, 4, burgers.LB, burgers.UB)
    loss_fcn = riemann.make_riemann_loss(model, ul, ur, nu, 10000, 1000, 200, weights, smooth_ic=True)

    t0 = time.perf_counter()
    history = train(loss_fcn, model.parameters(), adam_iters=2000, lr=1e-3, lbfgs_iters=2000, log_freq=500)

    t_train = time.perf_counter() - t0

    xt = np.stack([x, np.full_like(x, 0.5)], axis=1)
    with torch.no_grad():
        u_pinn = model(torch.tensor(xt, dtype=torch.get_default_dtype())).numpy()[:, 0]

    u_fan =  riemann.riemann_ext(x, [0.5], ul, ur)[0]
    l1_fan = np.trapezoid(np.abs(u_pinn - u_fan), x)

    dx = x[1] - x[0]
    max_slp = np.diff(u_pinn).max() / dx
    fan_slp = 1.0 / 0.5
    slp_ratio = max_slp / fan_slp

    results.append({"nu": nu, "l1_fan": l1_fan, "slope ratio": slp_ratio, "final_loss": history[-1]["loss"], "time_s": t_train})
    Path("scratch/rarefaction.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), f"scratch/rarefaction_{nu:.0e}.pt")

    print(f"nu {nu:.0e}: slope ratio {slp_ratio:.3f} final loss {history[-1]["loss"]:.3f} time ({t_train:.0f} s)")
