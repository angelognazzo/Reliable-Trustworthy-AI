import torch
from settings import VERBOSE
import torch.nn.functional as F
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from utils import tight_bounds
from backsubstitution import backsubstitution

class DeepPolyResnetBlock(torch.nn.Module):
    """
    Class implementing the ResNet block of the DeepPoly algorithm
    """
    
    # create a list of layers from the path
    def parse_paths(self, path):
        path = list(path)
        layers = []
        for p in path:
            print(type(p))
            if type(p) == torch.nn.modules.Conv2d:
                layers.append(DeepPolyConvolutionalLayer(p))
            elif type(p) == torch.nn.modules.activation.ReLU:
                layers.append(DeepPolyReluLayer(p))
            elif type(p) == torch.nn.modules.Identity:
                layers.append(DeepPolyIdentityLayer(p))
            else:
                raise Exception("Unknown layer type")
            
        return layers
    
    def __init__(self, l, prev_layers):
        super().__init__()

        self.block = l
        self.path_a = self.parse_paths(l.path_a)
        self.path_b = self.parse_paths(l.path_b)

        self.prev_layers = prev_layers
    
    def backsubstitute(self, layers, i, x, current_lower_bound, current_upper_bound, first_lower_bound, first_upper_bound):
        l = layers[i]
            
        if VERBOSE:
            print("DeepPolyNetwork: Performing backsubstitution")

        # perform backsubstitution
        lower_bound_tmp, upper_bound_tmp = backsubstitution(layers, i, x.shape[1], first_lower_bound, first_upper_bound)

        # dimensions should be preserved after backsubstitution
        assert lower_bound_tmp.shape == current_lower_bound.shape
        assert upper_bound_tmp.shape == current_upper_bound.shape

        # update the lower and upper bounds
        lower_bound_tighten, upper_bound_tighten = tight_bounds(
            current_lower_bound, current_upper_bound, lower_bound_tmp, upper_bound_tmp)

        # the correct order now should be respected
        assert (lower_bound_tighten <= upper_bound_tighten).all(
        ), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"
        
        return lower_bound_tighten, upper_bound_tighten
        
    

        
    def forward(self, x, lower_bound, upper_bound, input_shape, first_lower_bound, first_upper_bound):
        # compute the forward of the first path
        if VERBOSE:
            print("Forward pass: path a")
        x_a, lower_bound_a, upper_bound_a, input_shape_a = torch.clone(x), torch.clone(lower_bound), torch.clone(upper_bound), input_shape
        previous_layers_a = self.prev_layers + self.path_a
        for i, layer in enumerate(self.path_a):
            x_a, lower_bound_a, upper_bound_a, input_shape_a = layer(x_a, lower_bound_a, upper_bound_a, input_shape_a)
            if (type(layer) == DeepPolyLinearLayer or type(layer) == DeepPolyConvolutionalLayer) and i > 1:
                lower_bound_a, upper_bound_a = self.backsubstitute(previous_layers_a, i + len(self.prev_layers), x_a, lower_bound_a, upper_bound_a, first_lower_bound, first_upper_bound)
            
       
        # compute the forward of the second path
        if VERBOSE:
            print("Forward pass: path b")
        x_b, lower_bound_b, upper_bound_b, input_shape_b = torch.clone(x), torch.clone(lower_bound), torch.clone(upper_bound), input_shape
        previous_layers_b = self.prev_layers + self.path_b
        for i, layer in enumerate(self.path_b):
            x_b, lower_bound_b, upper_bound_b, input_shape_b = layer(x_b, lower_bound_b, upper_bound_b, input_shape_b)
            if (type(layer) == DeepPolyLinearLayer or type(layer) == DeepPolyConvolutionalLayer) and i > 1:
                lower_bound_b, upper_bound_b = self.backsubstitute(previous_layers_b, i + len(self.prev_layers), x_b, lower_bound_b, upper_bound_b, first_lower_bound, first_upper_bound)
       
        assert x_a.shape == x_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of input"
        assert lower_bound_a.shape == lower_bound_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of lower bound"
        assert upper_bound_a.shape == upper_bound_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of upper bound"
        assert input_shape_a == input_shape_b, "DeepPolyResnetBlock forward: the two paths have different shapes of input"
        
        new_x = x_a + x_b
        new_lower_bound = lower_bound_a + lower_bound_b
        new_upper_bound = upper_bound_a + upper_bound_b
        
        return new_x, new_lower_bound, new_upper_bound, input_shape_a
       
