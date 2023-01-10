import torch
from settings import VERBOSE
import torch.nn.functional as F
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer
from DeepPolyBatchNormLayer import DeepPolyBatchNormLayer
from utils import compute_out_dimension
#from backsubstitution import backsubstitution

class DeepPolyResnetBlock(torch.nn.Module):
    """
    Class implementing the ResNet block of the DeepPoly algorithm
    """
    
    # create a list of custom layers from the path
    def parse_paths(self, path, out_dimension, prev_layers):
        path = list(path)
        layers = []
        
        for p in path:
            out_dimension_tmp = compute_out_dimension(out_dimension, p)
            if type(p) == torch.nn.modules.Conv2d:
                layers.append(DeepPolyConvolutionalLayer(p, prev_layers.copy() + layers.copy(), out_dimension))
            elif type(p) == torch.nn.modules.activation.ReLU:
                layers.append(DeepPolyReluLayer(p, prev_layers.copy() + layers.copy(), out_dimension_tmp))
            elif type(p) == torch.nn.modules.Identity:
                layers.append(DeepPolyIdentityLayer(p, prev_layers.copy()+layers.copy(), out_dimension))
            elif type(p) == torch.nn.modules.BatchNorm2d:
                # TODO: FIX THIS
                layers.append(DeepPolyBatchNormLayer(p, prev_layers.copy()+layers.copy(), out_dimension))
            else:
                raise Exception("Unknown layer type")
            out_dimension = out_dimension_tmp
        return torch.nn.Sequential(*(layers)) # return a sequential layer with all the layers in the path, before + prev_layers
    
    def __init__(self, l, prev_layers, out_dimension):
        super().__init__()
        
        
        self.previous_layers = prev_layers
        self.out_dimension=out_dimension
        #self.input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        #self.input_shape_flatten = input_shape[0] * input_shape[1] * input_shape[2]
                
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None

        # complete paths all the way to the beginning of the network
        self.path_a = self.parse_paths(l.path_a, self.out_dimension, self.previous_layers)
        self.path_b = self.parse_paths(l.path_b, self.out_dimension, self.previous_layers)

        self.isRes=True

    def forward(self, lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag):
        # compute the forward of path_a
        if VERBOSE:
            print("Forward pass: path a")
            print(self.path_a)
        lower_bound_path_a=torch.clone(lower_bound)
        upper_bound_path_a=torch.clone(upper_bound)
        for l in self.path_a:
            lower_bound_path_a, upper_bound_path_a = l(lower_bound_path_a, upper_bound_path_a, first_lower_bound, first_upper_bound, flag=True)
        
        #_, _, lower_weights_cumulative_a, upper_weights_cumulative_a, lower_bias_cumulative_a, upper_bias_cumulative_a=backsubstitution(self.path_a, lower_bound, upper_bound)
        # compute the forward of path_b
        if VERBOSE:
            print("Forward pass: path b")
            print(self.path_b)
        lower_bound_path_b=torch.clone(lower_bound)
        upper_bound_path_b=torch.clone(upper_bound)
        for l in self.path_b:
            lower_bound_path_b, upper_bound_path_b = l(lower_bound_path_b, upper_bound_path_b, first_lower_bound, first_upper_bound, flag=True)

        #_, _, lower_weights_cumulative_b, upper_weights_cumulative_b, lower_bias_cumulative_b, upper_bias_cumulative_b=backsubstitution(self.path_b, lower_bound, upper_bound)
       
        #assert lower_bound_path_a.shape == lower_bound_path_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of lower bound"
        #assert upper_bound_path_a.shape == upper_bound_path_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of upper bound"
        
        lower_bound_to_return = lower_bound_path_a + lower_bound_path_b
        upper_bound_to_return = upper_bound_path_a + upper_bound_path_b
        
        # TODO: we need to compute the weight matrix here!!!
        #self.lower_weights = lower_weights_cumulative_a + lower_weights_cumulative_b
        #self.upper_weights = upper_weights_cumulative_a + upper_weights_cumulative_b
        
        #self.lower_bias = lower_bias_cumulative_a + lower_bias_cumulative_b
        #self.upper_bias = upper_bias_cumulative_a + upper_bias_cumulative_b
        
        return lower_bound_to_return, upper_bound_to_return
