import torch
from settings import VERBOSE

class DeepPolyIdentityLayer(torch.nn.Module):
    """
    Class implementing the identity layer of the DeepPoly algorithm
    """
    
    def __init__(self, l):
        super().__init__()
        
        self.layer = l
        
    def forward(self, x, lower_bound, upper_bound, input_shape):
        return x, lower_bound, upper_bound, input_shape