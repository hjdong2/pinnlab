
import numpy as np
import torch


def grad(y, x):

    dy = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]
    return dy

def rel_l2(pred, ref):
    """Relative L2 error ||pred - ref|| / ||ref|| """
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)
