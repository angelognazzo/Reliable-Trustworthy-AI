import torch
from settings import VERBOSE
import torch.nn.functional as F
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer

class DeepPolyResnetBlock(torch.nn.Module):
    """
    Class implementing the ResNet block of the DeepPoly algorithm
    """
    
    # create a list of layers from the path
    def parse_paths(self, paths):
        paths = list(paths)
        layers = []
        for p in paths:
            if type(p) == torch.nn.modules.Conv2d:
                layers.append(DeepPolyConvolutionalLayer(p))
            elif type(p) == torch.nn.modules.activation.ReLU:
                layers.append(DeepPolyReluLayer(p))
            elif type(p) == torch.nn.modules.Identity:
                layers.append(DeepPolyIdentityLayer(p))
            else:
                raise Exception("Unknown layer type")
        
        return layers
            
        
    def __init__(self, l):
        super().__init__()
        
        self.block = l
        self.path_a = self.parse_paths(l.path_a)
        self.path_b = self.parse_paths(l.path_b)
        
        self.weights = None
        self.bias = None
        
    def forward(self, x, lower_bound, upper_bound, input_shape):
        # compute the forward of the first path
        x_a, lower_bound_a, upper_bound_a, input_shape_a = x, lower_bound, upper_bound, input_shape
        for layer in self.path_a:
           x_a, lower_bound_a, upper_bound_a, input_shape_a = layer(x_a, lower_bound_a, upper_bound_a, input_shape_a)
        
        # compute the forward of the second path
        x_b, lower_bound_b, upper_bound_b, input_shape_b = x, lower_bound, upper_bound, input_shape
        for layer in self.path_b:
           x_b, lower_bound_b, upper_bound_b, input_shape_b = layer(x_b, lower_bound_b, upper_bound_b, input_shape_b)
           
        
        # TODO 
        # AGGREGATE RESULTS FOR X AND BOUNDS
        # COMPUTE WEIGHTS AND BIAS
