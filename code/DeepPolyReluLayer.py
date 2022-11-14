import torch
from settings import VERBOSE

# ! TODO: implement
class DeepPolyReluLayer(torch.nn.Module):

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, x, lower, upper):
        if VERBOSE:
            print("DeepPolyReluLayer: lower_bound shape %s, upper_bound shape %s, x shape %s",
                  lower.shape, upper.shape, x.shape)

        assert lower.shape == upper.shape

        return x, lower, upper
