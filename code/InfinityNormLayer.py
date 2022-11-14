import torch
from settings import VERBOSE

# return the lower and upper bounds of the infinity norm of the input image
class InfinityNormLayer(torch.nn.Module):

    def __init__(self, eps) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # lower = torch.clamp(inputs - pert, min=0.0).to(DEVICE)
        # upper = torch.clamp(inputs + pert, max=1.0).to(DEVICE)
        lower = torch.maximum(x - self.eps, torch.tensor(0))
        upper = torch.minimum(x + self.eps, torch.tensor(1))

        if VERBOSE:
            print("InfinityNormLayer: lower_bound shape %s, upper_bound shape %s", lower.shape, upper.shape)

        assert lower.shape == upper.shape
        assert lower.shape[0] == 1

        # return torch.flatten(lower), torch.flatten(upper)
        return lower, upper
