import torch
from settings import VERBOSE
import torch.nn.functional as F
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from utils import tight_bounds

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
            
    # perform backsubstitution to compute tighter lower and upper bounds
    def backsubstitution(self, layers, input_size, lower_bound, upper_bound):
        # initialize new lower and upper weights
        lower_weights = torch.eye(input_size)
        lower_bias = torch.zeros(1, input_size)
        upper_weights = torch.eye(input_size)
        upper_bias = torch.zeros(1, input_size)

        # ! TODO: use torch.where instead of for loops
        # iterate through the layers in reverse order starting from the layer before the current_layer to and excluding the first layer (InfinityNormLayer)
        for i in range(len(layers) - 1, -1, -1):

            if VERBOSE:
                print("Backsubstitution Loop: layer %s out of %s layers" % (i + 1, len(layers)))

            # get the current layer and its type
            layer = layers[i]
            layer_type = type(layer) == DeepPolyReluLayer
            
            # if a linear layer or convolutional is encountered get the actual weights and bias of the layer,
            # else (RELU layer) use the computed weight bounds
            upper_weights_tmp = layer.upper_weights if layer_type else layer.weights
            upper_bias_tmp = layer.upper_bias if layer_type else layer.bias
            lower_weights_tmp = layer.lower_weights if layer_type else layer.weights
            lower_bias_tmp = layer.lower_bias if layer_type else layer.bias

            upper_bias += torch.matmul(upper_bias_tmp, upper_weights)
            lower_bias += torch.matmul(lower_bias_tmp, lower_weights)
            upper_weights = torch.matmul(upper_weights_tmp, upper_weights)
            lower_weights = torch.matmul(lower_weights_tmp, lower_weights)

        # perform a new forward pass with the new weights to compute the new lower and upper bounds
        new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(
            lower_bound, upper_bound, lower_weights, lower_bias)
        _, new_upper_bound = DeepPolyLinearLayer.swap_and_forward(
            lower_bound, upper_bound, upper_weights, upper_bias)

        assert new_lower_bound.shape == new_upper_bound.shape, "DeepPolyResnetBlock backsubstitution: Error with the shape of the bounds"

        return new_lower_bound, new_upper_bound, lower_weights, upper_weights, lower_bias, upper_bias
    
    def __init__(self, l):
        super().__init__()
        
        self.block = l
        self.path_a = self.parse_paths(l.path_a)
        self.path_b = self.parse_paths(l.path_b)
        
        self.weights = None
        self.bias = None
        
    def forward(self, x, lower_bound, upper_bound, input_shape):
        # compute the forward of the first path
        if VERBOSE:
            print("Forward pass: path a")
        x_a, lower_bound_a, upper_bound_a, input_shape_a = x, lower_bound, upper_bound, input_shape
        for layer in self.path_a:
            x_a, lower_bound_a, upper_bound_a, input_shape_a = layer(x_a, lower_bound_a, upper_bound_a, input_shape_a)
        # compute the forward of the second path
        if VERBOSE:
            print("Forward pass: path b")
        x_b, lower_bound_b, upper_bound_b, input_shape_b = x, lower_bound, upper_bound, input_shape
        for layer in self.path_b:
            x_b, lower_bound_b, upper_bound_b, input_shape_b = layer(x_b, lower_bound_b, upper_bound_b, input_shape_b)
        
        assert x_a.shape == x_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of input"
        assert lower_bound_a.shape == lower_bound_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of lower bound"
        assert upper_bound_a.shape == upper_bound_b.shape, "DeepPolyResnetBlock forward: the two paths have different shapes of upper bound"
        assert input_shape_a == input_shape_b, "DeepPolyResnetBlock forward: the two paths have different shapes of input"
        
        lower_bound_a_tmp, upper_bound_a_tmp, lower_weights_a, upper_weights_a, lower_bias_a, upper_bias_a = self.backsubstitution(
            self.path_a, input_shape_a.numel(), lower_bound, upper_bound)
        lower_bound_b_tmp, upper_bound_b_tmp, lower_weights_b, upper_weights_b, lower_bias_b, upper_bias_b = self.backsubstitution(
            self.path_b, input_shape_b.numel(), lower_bound, upper_bound)
        
        # lower_bound_a, upper_bound_a = tight_bounds(lower_bound_a, upper_bound_a, lower_bound_a_tmp, upper_bound_a_tmp)
        # lower_bound_b, upper_bound_b = tight_bounds(lower_bound_b, upper_bound_b, lower_bound_b_tmp, upper_bound_b_tmp)
        
        # # sum the two paths
        # new_lower_bound = torch.add(lower_bound_a, lower_bound_b)
        # new_upper_bound = torch.add(upper_bound_a, upper_bound_b)
        # new_x = torch.add(x_a, x_b)
        
        # sum the two paths
        new_lower_bound = torch.add(lower_bound_a, lower_bound_b)
        new_upper_bound = torch.add(upper_bound_a, upper_bound_b)
        new_x = torch.add(x_a, x_b)
        
        new_lower_bound_tmp = torch.add(lower_bound_a_tmp, lower_bound_b_tmp)
        new_upper_bound_tmp = torch.add(upper_bound_a_tmp, upper_bound_b_tmp)
        
        new_lower_bound, new_upper_bound = tight_bounds(new_lower_bound, new_upper_bound, new_lower_bound_tmp, new_upper_bound_tmp)

        # compute the weights and bias of the current layer
        self.lower_weights = torch.add(lower_weights_a, lower_weights_b)
        self.upper_weights = torch.add(upper_weights_a, upper_weights_b)
        self.lower_bias = torch.add(lower_bias_a, lower_bias_b)
        self.upper_bias = torch.add(upper_bias_a, upper_bias_b)

        return new_x, new_lower_bound, new_upper_bound, input_shape_a
