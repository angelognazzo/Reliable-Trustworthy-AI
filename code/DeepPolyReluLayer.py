import torch
from settings import VERBOSE


class DeepPolyReluLayer(torch.nn.Module):
    """
    Class implementing the ReluLayer of the DeepPoly algorithm
    """

    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None
        
    def compute_weights(self):
        pass

    def forward(self, x, lower, upper):
        
        if VERBOSE:
            print("DeepPolyReluLayer forward: lower shape %s, upper shape %s" % (str(lower.shape), str(upper.shape)))
        
        assert lower.shape == upper.shape, "DeepPolyReluLayer forward: lower and upper bounds have different shapes"
        
        # compute the relu on the input
        x = self.layer(x)
        
        # all the points are negative
        if (upper <= 0).all():
            lower = torch.zeros_like(lower)
            upper = torch.zeros_like(upper)
            self.lower_weights = torch.zeros(lower.shape[1])
            self.upper_weights = torch.zeros(lower.shape[1])
            self.lower_bias = torch.zeros_like(lower)
            self.upper_bias = torch.zeros_like(upper)  
        # all the points are positive
        elif (lower >= 0).all():
            lower = upper
            self.lower_weights = torch.eye(lower.shape[1])
            self.upper_weights = torch.eye(lower.shape[1])
            self.lower_bias = torch.zeros_like(lower)
            self.upper_bias = torch.zeros_like(upper)
        # some points are negative and some are positive
        else:
            # TODO: optimize alpha with gradient descent
            alpha = 1.0
            # TODO: check the shape of eye
            self.lower_weights = alpha * torch.eye(lower.shape[1])
            slope = torch.div(upper, upper - lower)
            self.upper_weights = torch.diag(slope.squeeze())
            
            self.lower_bias = torch.zeros_like(lower)
            self.upper_bias = slope * lower # element-wise multiplication
            
            lower = torch.matmul(lower, self.lower_weights)
            upper = torch.matmul(upper, self.upper_weights)
            
        if VERBOSE:
            print("DeepPolyReluLayer forward: slope shape %s, self.lower_weights shape %s, self.upper_weights shape %s" %
                  (str(slope.shape), str(self.lower_weights.shape), str(self.upper_weights.shape)))
            print("DeepPolyReluLayer: lower_bound shape %s, upper_bound shape %s, x shape %s" %
                  (str(lower.shape), str(upper.shape), str(x.shape)))
            
        
        assert lower.shape == upper.shape
        assert self.lower_bias.shape == self.upper_bias.shape
        assert self.lower_weights.shape == self.upper_weights.shape
        assert (lower <= upper).all()

        return x, lower, upper
