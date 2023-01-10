import torch
from settings import VERBOSE


class InfinityNormLayer(torch.nn.Module):
    """
    Class implementing the LinearLayer of the DeepPoly algorithm
    Return the lower and upper bounds of the infinity norm of the input image
    """

    def __init__(self, eps) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # perturb the input, making sure not to go below or above 0 (pixels are defined between 0 and 1)
        lower = torch.maximum(x - self.eps, torch.tensor(0))
        upper = torch.minimum(x + self.eps, torch.tensor(1))

        if VERBOSE:
            print("InfinityNormLayer: lower_bound shape %s, upper_bound shape %s" % (lower.shape, upper.shape))

        #assert lower.shape == upper.shape
        #assert lower.shape[0] == 1
        #assert (lower <= upper).all(), "InfinityNormLayer: error with the box bounds: lower > upper"
       
        return lower, upper
