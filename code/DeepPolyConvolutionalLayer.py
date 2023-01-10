import torch
from settings import VERBOSE
import torch.nn.functional as F
from backsubstitution import backsubstitution
#from math import sqrt
#import numpy as np


class DeepPolyConvolutionalLayer(torch.nn.Module):
    """
    Class implementing the ConvolutionalLayer of the DeepPoly algorithm
    """
    
    def __init__(self, layer, previous_layers, input_shape) -> None:
        super().__init__()
        self.previous_layers = previous_layers
        self.input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        self.input_shape_flatten = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.isRes=False
        
        self.kernel = layer.weight.detach()
        if layer.bias is None:
            self.bias_kernel = None
        else:
            self.bias_kernel = layer.bias.detach()
        self.stride = layer.stride
        self.padding = layer.padding 
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer: weights shape: %s, stride %s, padding %s" % (
                str(self.kernel.shape), str(self.stride), str(self.padding)))

    def forward(self, lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag):

        if self.bias_kernel is None:
            self.bias_kernel = torch.zeros(self.kernel.shape[0])
       
        w = torch.eye(self.input_shape_flatten).view(list(self.input_shape) + [self.input_shape_flatten])
        w = w.permute(0, 1, 4, 2, 3)
        w = F.conv3d(w, self.kernel.unsqueeze(2), stride=tuple(
            [1] + list(self.stride)), padding=tuple([0] + list(self.padding))).permute(0, 1, 3, 4, 2)
        
        # remove the first empty dimension
        w = w[0]
        weights = torch.flatten(w, start_dim=0, end_dim=2).t()
        self.lower_weights = weights
        self.upper_weights = weights

        b = torch.ones(w.shape[:-1]) * self.bias_kernel[:, None, None]
        bias = torch.flatten(b).reshape(1,-1)
        self.lower_bias = bias
        self.upper_bias = bias
        
        if flag==True:
            lower_bound, upper_bound, _, _, _, _= backsubstitution(self.previous_layers + [self], first_lower_bound, first_upper_bound)
        else:
            bounds_upper=torch.empty_like(self.upper_bias)
            bounds_lower=torch.empty_like(self.lower_bias)
            upper_bound=bounds_upper.fill_( float("Inf"))
            lower_bound=bounds_lower.fill_( -float("Inf"))

        #assert lower_bound.shape == upper_bound.shape, "swap_and_forward CNN: lower and upper bounds have different shapes"
          
        return lower_bound, upper_bound










