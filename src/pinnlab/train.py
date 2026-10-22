
import time

import torch


def train(loss_fcn, params, adam_iters, lr, lbfgs_iters, log_freq=100, callback=None):
    """Adam to get a good start, and then turn to L-BFGS.
        loss_fcn() return (l_tot, parts)
        Return the history.
    """

    params = list(params)
    history = []
    t0 = time.perf_counter()
    it = 0

    def log(stage, loss, parts):
        elapsed = time.perf_counter() - t0
        row = {"stage": stage, "iter": it, "time": elapsed, "loss": loss.item(), **parts}
        if callback is not None:
            row.update(callback())

        history.append(row)
        if it % log_freq == 0:
            print(f"iter {it:5d}    loss {loss.item():.3e}    ({elapsed:.1f} s)")

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(adam_iters):
        opt.zero_grad()
        l_tot, adam_parts = loss_fcn()
        l_tot.backward()
        opt.step()

        if it % log_freq == 0:
            log("adam", l_tot, adam_parts)

        it += 1

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters,
                                  max_eval=int(1.25*lbfgs_iters), history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

        def closure():
            nonlocal it
            lbfgs.zero_grad()
            l_tot, lbfgs_parts = loss_fcn()
            l_tot.backward()

            if it % log_freq == 0:
                log("lbfgs", l_tot, lbfgs_parts)

            it += 1
            return l_tot

        lbfgs.step(closure)


    l_fnl, fnl_parts = loss_fcn()
    print(f"final loss {l_fnl.item():.3e}    total time {time.perf_counter() - t0:.1f} s")
    log("final", l_fnl, fnl_parts)
    return history
