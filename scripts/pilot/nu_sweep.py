
import json
import time
from pathlib import Path

import numpy as np
import torch

from pinnlab.models import MLP
from pinnlab.problems import burgers, riemann
from pinnlab.train import train

torch.set_num_threads(2)
ul, ur = 1.0, 0.0
s = (ul + ur) / 2
x = np.linspace(-1, 1, 8001)

weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
results = []

for nu in [1.0e-2, 3.0e-3, 1.0e-3]:
    torch.manual_seed(0)
    model = MLP(2, 1, 20, 4, burgers.LB, burgers.UB)
    loss_fcn = riemann.make_riemann_loss(model, ul, ur, nu, 10000, 1000, 200, weights, smooth_ic=True)

    t0 = time.perf_counter()
    history = train(loss_fcn, model.parameters(), adam_iters=2000, lr=1e-3, lbfgs_iters=2000, log_freq=500)

    t_train = time.perf_counter() - t0

    xt = np.stack([x, np.full_like(x, 0.5)], axis=1)
    with torch.no_grad():
        u_pinn = model(torch.tensor(xt, dtype=torch.get_default_dtype())).numpy()[:, 0]

    u_inv =  riemann.riemann_ext(x, [0.5], ul, ur)[0]
    u_vis = riemann.viscous_shk(x, [0.5], ul, ur, nu)[0]
    l1_inv = np.trapezoid(np.abs(u_pinn - u_inv), x)
    l1_vis = np.trapezoid(np.abs(u_pinn - u_vis), x)
    l1_thry = 4 * np.log(2) * nu
    posoff = riemann.shock_pos(x, u_pinn, ul, ur) - s * 0.5
    thick_ext = 2 * np.arctanh(0.8) * 4 * nu / (ul - ur)
    thick_ratio = riemann.shock_thickness(x, u_pinn, ul, ur) / thick_ext

    results.append({"nu": nu, "l1_inv": l1_inv, "l1_vis": l1_vis, "l1_thry": l1_thry, "offset": posoff, "thick_ratio": thick_ratio,
                    "final_loss": history[-1]["loss"], "time_s": t_train})
    Path("scratch/nu_sweep.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), f"scratch/nu_sweep_{nu:.0e}.pt")

    print(f"nu {nu:.0e}: L1 inv {l1_inv:.2e}  L1 vis {l1_vis:.2e} theory {l1_thry:.2e} "
          f"offset {posoff:+.2e} thickness ratio {thick_ratio:.3f} ({t_train:.0f} s)")
