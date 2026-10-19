
import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, width, depth, lb, ub, activation=nn.Tanh):
        super().__init__()

        lb_tsr = torch.as_tensor(lb, dtype=torch.get_default_dtype())
        ub_tsr = torch.as_tensor(ub, dtype=torch.get_default_dtype())

        self.register_buffer("lb", lb_tsr)
        self.register_buffer("ub", ub_tsr)

        # build the list of layers
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(activation())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(activation())

        layers.append(nn.Linear(width, out_dim))

        self.net = nn.Sequential(*layers)

        # loop over layers; for each nn.Linear, apply Xavier-normal
        # into to .weight, and zeros to .bias
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def normalize(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, x):
        # x has shape (N, in_dim)
        # normalize x to [-1, 1] using self.lb / self.ub
        x = self.normalize(x)
        return self.net(x)
