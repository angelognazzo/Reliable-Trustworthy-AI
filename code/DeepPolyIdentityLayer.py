import torch
from settings import VERBOSE

class DeepPolyIdentityLayer(torch.nn.Module):
    """
    Class implementing the identity layer of the DeepPoly algorithm
    """
    
    def __init__(self, layer, previous_layers, input_shape):
        super().__init__()
        
        self.input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        self.input_shape_flatten = input_shape[0] * input_shape[1] * input_shape[2]
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None
        
        self.isRes=False
        
    def forward(self, lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag):
        if VERBOSE:
            print("DeepPolyIdentityLayer: lower_bound shape %s, upper_bound shape %s" % (
                str(lower_bound.shape), str(upper_bound.shape)))
        self.lower_weights = torch.eye(self.input_shape_flatten)
        self.upper_weights = torch.eye(self.input_shape_flatten)
        self.lower_bias = torch.zeros(1, self.input_shape_flatten)
        self.upper_bias = torch.zeros(1, self.input_shape_flatten)
       
        return lower_bound, upper_bound