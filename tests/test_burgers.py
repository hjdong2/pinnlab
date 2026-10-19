
import numpy as np
import pytest
import torch

from pinnlab import train
from pinnlab.models import MLP
from pinnlab.problems import burgers


def test_residual_mms():

    nu, lam1 = 0.07, 1.3
    random = torch.rand(50, 2, dtype=torch.float64)
    lb = torch.tensor(burgers.LB, dtype=torch.float64)
    ub = torch.tensor(burgers.UB, dtype=torch.float64)
    xt = lb + (ub - lb) * random

    def u_fcn(xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return torch.sin(np.pi * x) * torch.exp(-t)

    def u_t(xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return -torch.sin(np.pi * x) * torch.exp(-t)

    def u_x(xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return np.pi * torch.cos(np.pi * x) * torch.exp(-t)

    def u_xx(xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return -np.pi*np.pi * torch.sin(np.pi * x) * torch.exp(-t)

    res_ext = u_t(xt) + lam1 * u_fcn(xt) * u_x(xt) - nu * u_xx(xt)
    res_num = burgers.residual(u_fcn, xt, nu, lam1)

    torch.testing.assert_close(res_num, res_ext)


def test_reference_ic():
    # test the reference soln matches with IC
    x = np.linspace(-1, 1, 101)
    u_ext0 = -np.sin(np.pi * x)

    u_0 = burgers.reference_soln(x, 0.0)
    u_0 = u_0.reshape(-1)

    u_1 = burgers.reference_soln(x, 1e-6)
    u_1 = u_1.reshape(-1)

    np.testing.assert_allclose(u_0, u_ext0, rtol=0, atol=0)
    np.testing.assert_allclose(u_1, u_ext0, rtol=1e-4, atol=1e-4)


def test_reference_sym_bcs():
    # test odd symmetry and boundary values
    x = np.linspace(-1, 1, 101)
    u = burgers.reference_soln(x, [0.1, 0.5, 1.0])

    uodd = -burgers.reference_soln(-x, [0.1, 0.5, 1.0])

    np.testing.assert_allclose(uodd, u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u[:, [0, -1]], 0.0, atol=1e-12)


def test_reference_basdevant():
    # test against Basdevant results
    h  = 1e-6
    ts = np.linspace(0.49, 0.53, 81)

    x = np.array([-h, h])
    u = burgers.reference_soln(x, ts)

    slp = np.abs((u[:, -1] - u[:, 0])) / (2 * h)
    slpmax = slp.max()
    tsmax = ts[np.argmax(slp)]

    assert slpmax == pytest.approx(152.00516, abs=1e-4)
    assert tsmax == pytest.approx(0.5105, abs=1e-3)


def test_forward_loss_basic():
    torch.manual_seed(0)
    model = MLP(2, 1, 16, 3, burgers.LB, burgers.UB)
    weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
    loss_fcn = burgers.make_forward_loss(model, burgers.NU, 500, 64, 64, weights)

    l_tot, parts = loss_fcn()

    assert l_tot.dim() == 0
    assert torch.isfinite(l_tot)
    assert set(parts) == {"pde", "ic", "bc"}
    l_tst = 0.0
    for name, value in parts.items():
        assert value >= 0.0
        l_tst += weights[name] * value

    assert l_tot.item() == pytest.approx(l_tst, rel=1e-6)

    l_tot1, parts1 = loss_fcn()
    assert l_tot.item() == l_tot1.item()


def test_forward_loss_grad():
    for term in ["pde", "ic", "bc"]:
        torch.manual_seed(0)
        model = MLP(2, 1, 16, 3, burgers.LB, burgers.UB)
        weights = {"pde": 0.0, "ic": 0.0, "bc": 0.0}
        weights[term] = 1.0
        loss_fcn = burgers.make_forward_loss(model, burgers.NU, 500, 64, 64, weights)

        l_tot, parts = loss_fcn()
        l_tot.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{term}: no gradient for {name}"


def test_train_loss():
    torch.manual_seed(0)
    model = MLP(2, 1, 16, 3, burgers.LB, burgers.UB)
    weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
    loss_fcn = burgers.make_forward_loss(model, burgers.NU, 500, 64, 64, weights)

    history = train.train(loss_fcn, model.parameters(), adam_iters=200, lr=1e-3, lbfgs_iters=50, log_freq=10)

    its = [h["iter"] for h in history]

    assert history[-1]["stage"] == "final"
    assert history[-1]["loss"] <= 0.5 * history[0]["loss"]
    assert any(h["stage"] == "lbfgs" for h in history)
    assert any(h["stage"] == "adam" for h in history)
    assert all(a <= b for a, b in zip(its, its[1:]))


def test_train_adam_only():
    torch.manual_seed(0)
    model = MLP(2, 1, 16, 3, burgers.LB, burgers.UB)
    weights = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
    loss_fcn = burgers.make_forward_loss(model, burgers.NU, 500, 64, 64, weights)

    history = train.train(loss_fcn, model.parameters(), adam_iters=200, lr=1e-3, lbfgs_iters=0, log_freq=10)

    its = [h["iter"] for h in history]

    assert history[-1]["stage"] == "final"
    assert history[-1]["loss"] <= 0.5 * history[0]["loss"]
    assert not any(h["stage"] == "lbfgs" for h in history)
    assert any(h["stage"] == "adam" for h in history)
    assert all(a <= b for a, b in zip(its, its[1:]))


def test_coeffs():
    coeffs = burgers.BurgersCoeffs(0.5, 0.05)

    assert coeffs.lambda1.item() == pytest.approx(0.5, abs=1.0e-4)
    assert coeffs.nu.item() == pytest.approx(0.05, abs=1.0e-4)

    assert {name for name, p in coeffs.named_parameters()} == {"lambda1", "log_nu"}

    with torch.no_grad():
        coeffs.log_nu.fill_(-50.0)

    assert coeffs.nu.item() > 0.0

    coeffs = burgers.BurgersCoeffs(0.5, 0.05)
    (2 * coeffs.nu).backward()

    assert coeffs.log_nu.grad.item() == pytest.approx(2 * coeffs.nu.item(), rel=1.0e-6)


def test_inverse_loss():
    torch.manual_seed(0)
    model = MLP(2, 1, 16, 3, burgers.LB, burgers.UB)
    coeffs = burgers.BurgersCoeffs(0.5, 0.05)
    x, t, u_ref = burgers.reference_grid(nx=64, nt=21)
    loss_fcn = burgers.make_inverse_loss(model, coeffs, x, t, u_ref, 200, 0.01, 500, {"data": 1.0, "pde": 1.0})

    l_tot, parts = loss_fcn()
    assert l_tot.dim() == 0
    assert torch.isfinite(l_tot)
    assert set(parts) == {"data", "pde"}

    l_tot.backward()

    for name, p in coeffs.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f" no gradient for {name}"


