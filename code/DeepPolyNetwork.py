import torch
from InfinityNormLayer import InfinityNormLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from settings import VERBOSE


class DeepPolyNetwork(torch.nn.Module):
    """
    Perform a DeepPoly analysis on the given network
    """
    def __init__(self, net, eps) -> None:
        super().__init__()

        self.eps = eps
        self.net = net

        self.layers = [InfinityNormLayer(self.eps)]
       
        # create a custom layer for each layer in the original network skipping the flattening and normalization layers (self.net[0] and self.net[1])
        for i in range(1, len(self.net.layers)):
            l = self.net.layers[i]
            
            # skip the flattening layer. This layer is not present in every network, For example in CNNs. That's why this check is necessary
            if type(l) == torch.nn.modules.flatten.Flatten:
                continue
            
            # create a custom layer based on the type of the original layer
            if type(l) == torch.nn.modules.linear.Linear:
                self.layers.append(DeepPolyLinearLayer(net, l))
            elif type(l) == torch.nn.modules.activation.ReLU:
                self.layers.append(DeepPolyReluLayer(l))
            elif type(l) == torch.nn.modules.Conv2d:
                self.layers.append(DeepPolyConvolutionalLayer(l))
            else:
                print("DeepPolyNetwork constructor ERROR: layer type not supported")

        if VERBOSE:
            print("DeepPolyNetwork: Created %s layers (Infinity norm layer included)" % (len(self.layers)))

        assert len(self.layers) > 0, "DeepPolyNetwork constructor: no layers created"
        assert len(self.layers) == len(self.net.layers) - 1, "DeepPolyNetwork constructor: number of layers mismatch compared to the original network"

        self.lower_bounds_list = []
        self.upper_bounds_list = []
        self.activation_list = []

    # perform backsubstitution to compute tighter lower and upper bounds
    def backsubstitution(self, current_layer, input_size):

        # get the first bounds
        first_lower_bound = self.lower_bounds_list[0]
        first_upper_bound = self.upper_bounds_list[0]
        # initialize new lower and upper weights
        lower_weights = torch.eye(input_size)
        lower_bias = torch.zeros(1, input_size)
        upper_weights = torch.eye(input_size)
        upper_bias = torch.zeros(1, input_size)

        # ! TODO: use torch.where instead of for loops
        # iterate through the layers in reverse order starting from the second to last layer (penultimum) to and excluding the first layer (InfinityNormLayer)
        for i in range(current_layer, 0, -1):

            if VERBOSE:
                print("Backsubstitution Loop: layer %s out of %s layers" % (i, current_layer))
                
            # get the current layer and its type
            layer = self.layers[i]
            layer_type = type(layer) == DeepPolyReluLayer
            # if a linear layer is encountered get the actual weights and bias of the layer, else use the computed weight bounds
            upper_weights_tmp = layer.upper_weights if layer_type else layer.weights
            upper_bias_tmp = layer.upper_bias if layer_type else layer.bias
            lower_weights_tmp = layer.lower_weights if layer_type else layer.weights
            lower_bias_tmp = layer.lower_bias if layer_type else layer.bias

            upper_bias += torch.matmul(upper_bias_tmp, upper_weights)
            lower_bias += torch.matmul(lower_bias_tmp, lower_weights)
            upper_weights = torch.matmul(upper_weights_tmp, upper_weights)
            lower_weights = torch.matmul(lower_weights_tmp, lower_weights)

        # perform a new forward pass with the new weights to compute the new lower and upper bounds
        new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, lower_weights, lower_bias)
        _, new_upper_bound = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, upper_weights, upper_bias)

        assert new_lower_bound.shape == new_upper_bound.shape
        # assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"

        return new_lower_bound, new_upper_bound

    def forward(self, x):
        # perturb the input image passing the input through the infinity norm custom layer

        lower_bound, upper_bound = self.layers[0](x)
        input_shape = x.shape
        
        # normalize the input image
        lower_bound = self.net.layers[0](lower_bound).flatten().reshape(1, -1)
        upper_bound = self.net.layers[0](upper_bound).flatten().reshape(1, -1)

        
        # if the first layer is a Flatten layer, the execute it
        # this check is necessary because in MLPs we have a Flatten layer, but not in CNNs
        if type(self.net.layers[1]) == torch.nn.modules.flatten.Flatten:
            lower_bound = self.net.layers[1](lower_bound)
            upper_bound = self.net.layers[1](upper_bound)
        
        # save the initial lower and upper bounds
        self.lower_bounds_list.append(lower_bound)
        self.upper_bounds_list.append(upper_bound)
       
        # pass x through normalization layers
        x = self.net.layers[0](x)
        x = x.flatten().reshape(1, -1)
        
        # same check as above
        if type(self.net.layers[1]) == torch.nn.modules.flatten.Flatten:
            x = self.net.layers[1](x)
        self.activation_list.append(x)
        
        # input dimensions should be the same even after our transformations
        assert x.shape == lower_bound.shape == upper_bound.shape, "DeepPolyNetwork forward: input shape mismatch after normalization and flattening"
        if VERBOSE:
            print("DeepPolyNetwork forward: shape after normalization and flattening: x: %s, lower bound %s, upper bound %s" % (x.shape, lower_bound.shape, upper_bound.shape))

        # perform the forward pass for each custom layer (skipping the infinity custom norm layer)
        for i in range(1, len(self.layers)):
            
            # get the current layer
            l = self.layers[i]

            if VERBOSE:
                print("DeepPolyNetwork: Forward pass for layer %s of type %s, out of %s layers" % (i + 1, type(l), len(self.layers)))

            # ! perform the FORWARD pass for the current layer
            x, lower_bound, upper_bound, input_shape = l(x, lower_bound, upper_bound, input_shape)
            
            if VERBOSE:
                print("DeepPolyNetwork forward: shape after layer %s: x: %s, lower bound %s, upper bound %s" % (i + 1, x.shape, lower_bound.shape, upper_bound.shape))
            assert x.shape == lower_bound.shape == upper_bound.shape, "DeepPolyNetwork forward: input shape mismatch after forward pass for layer %s" % (i)
            
            # if l not a RELU layer, perform backsubstitution
            if (type(l) == DeepPolyLinearLayer or type(l) == DeepPolyConvolutionalLayer) and i > 1:
                if VERBOSE:
                    print("DeepPolyNetwork: Performing backsubstitution")
                
                lower_bound_tmp, upper_bound_tmp = self.backsubstitution(i, x.shape[1])
                
                
                assert lower_bound_tmp.shape == lower_bound.shape
                assert upper_bound_tmp.shape == upper_bound.shape

                # get the tightest bound possible
                # when there is an interesection between the two bounds
                mask_positive = torch.max(lower_bound_tmp, lower_bound) <= torch.min(upper_bound_tmp, upper_bound)
                mask_negative = torch.logical_not(mask_positive)
                
                lower_bound = torch.where(mask_positive, torch.max(lower_bound_tmp, lower_bound), lower_bound)
                upper_bound =  torch.where(mask_positive, torch.min(upper_bound_tmp, upper_bound), upper_bound)
                
                assert (lower_bound <= upper_bound).all(), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"
                
                # when there is no intersection between the two bounds
                mask_tighter_disjoint = (upper_bound_tmp - lower_bound_tmp < upper_bound - lower_bound) & (upper_bound_tmp - lower_bound_tmp >= 0)
                lower_bound = torch.where(mask_negative & mask_tighter_disjoint, lower_bound_tmp, lower_bound)
                upper_bound = torch.where(mask_negative & mask_tighter_disjoint, upper_bound_tmp, upper_bound)
                
                assert (lower_bound <= upper_bound).all(), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"
                
            # save the newly computed bounds
            self.lower_bounds_list.append(lower_bound)
            self.upper_bounds_list.append(upper_bound)
            self.activation_list.append(x)

        if VERBOSE:
            print("DeepPolyNetwork: Forward pass completed")

        assert len(self.lower_bounds_list) == len(self.upper_bounds_list), "DeepPolyNetwork forward pass completed: Error: number of lower bounds != number of upper bounds"

        return x, self.lower_bounds_list, self.upper_bounds_list
