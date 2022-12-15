import torch
from settings import VERBOSE
import torch.nn.functional as F
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from DeepPolyBatchNormLayer import DeepPolyBatchNormLayer
from utils import tight_bounds, compute_out_dimension
from backsubstitution import backsubstitution
from backsubstitution import compute_new_weights_and_bias

class DeepPolyResnetBlock(torch.nn.Module):
    """
    Class implementing the ResNet block of the DeepPoly algorithm
    """
    
    # create a list of custom layers from the path
    def parse_paths(self, path, out_dimension):
        path = list(path)
        layers = []
        for p in path:
            out_dimension = compute_out_dimension(out_dimension, p)
            if type(p) == torch.nn.modules.Conv2d:
                layers.append(DeepPolyConvolutionalLayer(p))
            elif type(p) == torch.nn.modules.activation.ReLU:
                layers.append(DeepPolyReluLayer(p, out_dimension))
            elif type(p) == torch.nn.modules.Identity:
                layers.append(DeepPolyIdentityLayer(p))
            elif type(p) == torch.nn.modules.BatchNorm2d:
                layers.append(DeepPolyBatchNormLayer(p))
            else:
                raise Exception("Unknown layer type")
            
        return layers
    
    def __init__(self, l, prev_layers, out_dimension):
        super().__init__()

        self.block = l
        self.path_a = self.parse_paths(l.path_a, out_dimension)
        self.path_b = self.parse_paths(l.path_b, out_dimension)

        self.prev_layers = prev_layers
    
    def backsubstitute(self, layers, i, x, current_lower_bound, current_upper_bound, first_lower_bound, first_upper_bound):
            
        if VERBOSE:
            print("Resnet Block: Performing backsubstitution")

        # perform backsubstitution all the way to the begining of the NETWORK
        lower_bound_tmp, upper_bound_tmp = backsubstitution(layers, i, x.shape[1], first_lower_bound, first_upper_bound)

        # dimensions should be preserved after backsubstitution
        assert lower_bound_tmp.shape == current_lower_bound.shape
        assert upper_bound_tmp.shape == current_upper_bound.shape

        # tighten the bounds
        lower_bound_tighten, upper_bound_tighten = tight_bounds(
            current_lower_bound, current_upper_bound, lower_bound_tmp, upper_bound_tmp)

        # the correct order should be respected
        assert (lower_bound_tighten <= upper_bound_tighten).all(
        ), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"
        
        return lower_bound_tighten, upper_bound_tighten
        
        
    def forward(self, x, lower_bound, upper_bound, input_shape, first_lower_bound, first_upper_bound):
        # compute the forward of path_a
        if VERBOSE:
            print("Forward pass: path a")
        x_a, lower_bound_a, upper_bound_a, input_shape_a = torch.clone(x), torch.clone(lower_bound), torch.clone(upper_bound), input_shape
        # we need to compute the backsubstitution to the begining of the NETWORK, therefore we need to know which layers come before
        previous_layers_a = self.prev_layers + self.path_a
        for i, layer in enumerate(self.path_a):   
            
            # forward pass of current layer of path_a
            x_a, lower_bound_a, upper_bound_a, input_shape_a = layer(x_a, lower_bound_a, upper_bound_a, input_shape_a)
            
            # perform backsubstitution all the way to the begining of the NETWORK if needed
            if (type(layer) == DeepPolyLinearLayer or type(layer) == DeepPolyConvolutionalLayer) and i > 1:
                lower_bound_a, upper_bound_a = self.backsubstitute(previous_layers_a, i + len(self.prev_layers), x_a, lower_bound_a, upper_bound_a, first_lower_bound, first_upper_bound)
        
        # compute the forward of path_b
        if VERBOSE:
            print("Forward pass: path b")
        x_b, lower_bound_b, upper_bound_b, input_shape_b = torch.clone(x), torch.clone(lower_bound), torch.clone(upper_bound), input_shape
        # we need to compute the backsubstitution to the begining of the NETWORK, therefore we need to know which layers come before
        previous_layers_b = self.prev_layers + self.path_b
        for i, layer in enumerate(self.path_b):
            # forward pass of current layer of path_b
            x_b, lower_bound_b, upper_bound_b, input_shape_b = layer(x_b, lower_bound_b, upper_bound_b, input_shape_b)
                
            #  perform backsubstitution all the way to the begining of the NETWORK if needed
            if (type(layer) == DeepPolyLinearLayer or type(layer) == DeepPolyConvolutionalLayer) and i > 1:
                lower_bound_b, upper_bound_b = self.backsubstitute(previous_layers_b, i + len(self.prev_layers), x_b, lower_bound_b, upper_bound_b, first_lower_bound, first_upper_bound)

       
        assert x_a.shape == x_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of input"
        assert lower_bound_a.shape == lower_bound_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of lower bound"
        assert upper_bound_a.shape == upper_bound_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of upper bound"
        assert input_shape_a == input_shape_b, "DeepPolyResnetBlock forward: the two paths have different shapes of input"
        
        starting_lower_weights_a = previous_layers_a[-1].weights
        starting_upper_weights_a = previous_layers_a[-1].weights
        starting_lower_bias_a = previous_layers_a[-1].bias  # .reshape(1,-1)
        starting_upper_bias_a = previous_layers_a[-1].bias  # .reshape(1,-1)

        W_a_lower, W_a_upper, b_a_lower, b_a_upper = compute_new_weights_and_bias(
            previous_layers_a[1:-1], starting_lower_weights_a, starting_upper_weights_a, starting_lower_bias_a, starting_upper_bias_a)
        
        starting_lower_weights_b = previous_layers_b[-1].weights
        starting_upper_weights_b = previous_layers_b[-1].weights
        starting_lower_bias_b = previous_layers_b[-1].bias  # .reshape(1,-1)
        starting_upper_bias_b = previous_layers_b[-1].bias  # .reshape(1,-1)

        # qua il problema...
        W_b_lower, W_b_upper, b_b_lower, b_b_upper = compute_new_weights_and_bias(
            previous_layers_b[1:-1], starting_lower_weights_b, starting_upper_weights_b, starting_lower_bias_b, starting_upper_bias_b)

        W_tot_lower = W_a_lower + W_b_lower
        W_tot_upper = W_a_upper + W_b_upper
        b_tot_lower = b_a_lower + b_b_lower
        b_tot_upper = b_a_upper + b_b_upper
        
        new_lower_bound_tmp=torch.matmul(first_lower_bound, W_tot_lower)+b_tot_lower
        new_upper_bound_tmp=torch.matmul(first_upper_bound, W_tot_upper)+b_tot_upper

        # this is the alternative way to compute the bounds by optimizing the paths
        new_x = x_a + x_b
        new_lower_bound = lower_bound_a + lower_bound_b
        new_upper_bound = upper_bound_a + upper_bound_b
        
        new_lower_bound_tighten, new_upper_bound_tighten = tight_bounds(new_lower_bound, new_upper_bound,
                                                                        new_lower_bound_tmp, new_upper_bound_tmp)
        
        
        return new_x, new_lower_bound_tighten, new_upper_bound_tighten, input_shape_a
       
