
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from pinnlab.models import MLP
from pinnlab.plotting import plot_his, plot_soln
from pinnlab.problems import burgers
from pinnlab.train import train
from pinnlab.utils import rel_l2


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.quick:
        cfg["name"] += "_quick"
        cfg["train"].update(adam_iters=200, lbfgs_iters=100, log_freq=10)
        cfg["sampling"]["n_f"] = 1000

    torch.set_num_threads(args.threads)

    dtype = getattr(torch, cfg["dtype"])
    seed = cfg["seed"]
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)

    out = Path(args.out) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))

    x, t, u_ref = burgers.reference_grid(nu=cfg["nu"])

    model = MLP(2, 1, cfg["model"]["width"], cfg["model"]["depth"], burgers.LB, burgers.UB)
    if cfg["problem"] == "forward":
        loss_fcn = burgers.make_forward_loss(
            model, cfg["nu"],
            cfg["sampling"]["n_f"], cfg["sampling"]["n_0"], cfg["sampling"]["n_b"],
            cfg["loss_weights"], seed
        )
        params, callback, keys = model.parameters(), None, ["pde", "ic", "bc"]
        extra = None
    elif cfg["problem"] == "inverse":
        coeffs = burgers.BurgersCoeffs(cfg["lambda_init"], cfg["nu_init"])
        loss_fcn = burgers.make_inverse_loss(
            model, coeffs, x, t, u_ref,
            cfg["data"]["n_data"], cfg["data"]["noise"],
            cfg["sampling"]["n_f"], cfg["loss_weights"], seed
        )

        params = [*model.parameters(), *coeffs.parameters()]
        def callback():
            return {"lambda1": coeffs.lambda1.item(), "nu": coeffs.nu.item()}
        keys = ["data", "pde"]
        extra = {"lambda1": 1.0, "nu": cfg["nu"]}
    else:
        raise ValueError(f"unknown problem: {cfg['problem']!r}")

    history = train(loss_fcn, params, callback=callback, **cfg["train"])

    X, T = np.meshgrid(x, t)
    xt = np.stack([X.ravel(), T.ravel()], axis=1)
    xt_t = torch.tensor(xt, dtype=torch.get_default_dtype())
    with torch.no_grad():
        u_pred = model(xt_t).numpy().reshape(u_ref.shape)

    assert u_pred.shape == u_ref.shape
    err = rel_l2(u_pred, u_ref)
    print(f"relative L2 error: {err:.3e}")

    metrics = {
            "rel_l2": float(err),
            "final_loss": history[-1]["loss"],
            "train_time_s": history[-1]["time"],
            "iterations": history[-1]["iter"],
        }
    if cfg["problem"] == "inverse":
        learned = callback()
        metrics.update(learned)
        metrics["lambda1_err"] = abs(learned["lambda1"] - 1)
        metrics["nu_rel_err"] = abs(learned["nu"] - cfg["nu"]) / cfg["nu"]

    print(json.dumps(metrics, indent=2))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_soln(x, t, u_pred, u_ref, out / "solution.png")
    plot_his(history, keys, out / "loss.png", extra=extra)
    (out / "history.json").write_text(json.dumps(history))
    torch.save(model.state_dict(), out / "model.pt")
    print(f"wrote results to {out}/")


if __name__ == "__main__":
    main()
