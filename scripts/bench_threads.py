
import time

import torch

from pinnlab.models import MLP
from pinnlab.problems import burgers

for n in [1, 2, 4, 7, 14]:
    torch.set_num_threads(n)
    torch.manual_seed(0)
    model = MLP(2, 1, 20, 8, burgers.LB, burgers.UB)
    loss_fcn = burgers.make_forward_loss(model, burgers.NU, 10000, 256, 200, {"pde": 1.0, "ic": 1.0, "bc": 1.0})
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    for i in range(5):
        opt.zero_grad()
        l_tot, part = loss_fcn()
        l_tot.backward()
        opt.step()

    t0 = time.perf_counter()

    for i in range(50):
        opt.zero_grad()
        l_tot, part = loss_fcn()
        l_tot.backward()
        opt.step()

    print(f"threads {n:5d}  {time.perf_counter() - t0:.1f} s")
