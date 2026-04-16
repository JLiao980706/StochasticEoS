import torch
import torch.nn as nn


class FCTanh(nn.Module):
    """Fully-connected network with two hidden layers of 200 units and tanh activation.
    Follows the architecture in Cohen et al. [2021].
    Input: flattened CIFAR-10 images (3072,)
    Output: logits (10,)
    """

    def __init__(self, input_dim: int = 3072, hidden_dim: int = 200, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
