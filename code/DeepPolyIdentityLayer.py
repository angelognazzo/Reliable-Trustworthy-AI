import torch
from settings import VERBOSE

class DeepPolyIdentityLayer(torch.nn.Module):
    """
    Class implementing the identity layer of the DeepPoly algorithm
    """
    
    def __init__(self, l):
        super().__init__()
        
        self.layer = l
        self.weights = None
        self.bias = None
        
    def forward(self, x, lower_bound, upper_bound, input_shape):
        if VERBOSE:
            print("DeepPolyIdentityLayer: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))
        self.weights = torch.eye(input_shape.numel())
        self.bias = torch.zeros(1, input_shape.numel())
       
        return x, lower_bound, upper_bound, input_shape