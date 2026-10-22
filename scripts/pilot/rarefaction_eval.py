
import numpy as np
import torch

from pinnlab.models import MLP
from pinnlab.problems import burgers, riemann

ul, ur = 0.0, 1.0
x = np.linspace(-1, 1, 8001)
ts = [0.25, 0.5, 0.75]

def pinn_profile(model, x, tk):
    xt = np.stack([x, np.full_like(x, tk)], axis=1)
    with torch.no_grad():
        return model(torch.tensor(xt, dtype=torch.get_default_dtype())).numpy()[:, 0]


for nu in [1.0e-2, 1.0e-3]:
    model = MLP(2, 1, 20, 4, burgers.LB, burgers.UB)
    model.load_state_dict(torch.load(f"scratch/rarefaction_{nu:.0e}.pt"))

    for tk in ts:
        u = pinn_profile(model, x, tk)
        u_fan = riemann.riemann_ext(x, [tk], ul, ur)[0]
        l1 = np.trapezoid(np.abs(u - u_fan), x)
        print(f"nu {nu:.0e} tk {tk:.2f} L1 {l1:.2e}")
