import torch
from InfinityNormLayer import InfinityNormLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from DeepPolyReluLayer import DeepPolyReluLayer
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
        # create a custom layer for each layer in the original network skipping the flattening and normalization layers
        for i in range(2, len(self.net.layers)):
            l = self.net.layers[i]
            # TODO: add the convolution layer 
            if type(l) == torch.nn.modules.linear.Linear:
                self.layers.append(DeepPolyLinearLayer(net, l))
            elif type(l) == torch.nn.modules.activation.ReLU:
                self.layers.append(DeepPolyReluLayer(l))
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

        # iterate through the layers in reverse order starting from the second to last layer (penultimum) to and excluding the first layer (InfinityNormLayer)
        for i in range(current_layer, 0, -1):

            if VERBOSE:
                print("Backsubstitution Loop: layer %s out of %s layers" % (i, current_layer))
                
            # get the current layer and its type
            layer = self.layers[i]
            layer_type = type(layer) == DeepPolyReluLayer

            # if a linear layer is encountered get the actual weights and bias of the layer, else use the computed bounds
            upper_weights_tmp = layer.upper_weights if layer_type else layer.weights
            upper_bias_tmp = layer.upper_bias if layer_type else layer.bias
            lower_weights_tmp = layer.lower_weights if layer_type else layer.weights
            lower_bias_tmp = layer.lower_bias if layer_type else layer.bias
            
            upper_bias += torch.matmul(upper_bias_tmp, upper_weights)
            lower_bias += torch.matmul(lower_bias_tmp, lower_weights)
            upper_weights = torch.matmul(upper_weights_tmp, upper_weights)
            lower_weights = torch.matmul(lower_weights_tmp, lower_weights)

        # perform a new forward pass with the new weights to compute the new lower and upper bounds for the previous layer
        new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, lower_weights, lower_bias)
        _, new_upper_bound = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, upper_weights, upper_bias)
   

        assert new_lower_bound.shape == new_upper_bound.shape
        assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"

        return new_lower_bound, new_upper_bound

    def forward(self, x):
        # perturb the input image passing the input through the infinity norm layer
        lower_bound, upper_bound = self.layers[0](x)
        
        # normalize and flatten the input image
        lower_bound = self.net.layers[0](lower_bound)
        lower_bound = self.net.layers[1](lower_bound)
        upper_bound = self.net.layers[0](upper_bound)
        upper_bound = self.net.layers[1](upper_bound)
        # save the initial lower and upper bounds
        self.lower_bounds_list.append(lower_bound)
        self.upper_bounds_list.append(upper_bound)
       
        # pass x through the flatten and normalization layers
        x = self.net.layers[0](x)
        x = self.net.layers[1](x)
        self.activation_list.append(x)
        
        # input should be flattened and dimensions should be the same
        assert x.shape[0] == 1
        assert x.shape == lower_bound.shape == upper_bound.shape, "DeepPolyNetwork forward: input shape mismatch after normalization and flattening"
        if VERBOSE:
            print("DeepPolyNetwork forward: shape after normalization and flattening: x: %s, lower bound %s, upper bound %s" % (x.shape, lower_bound.shape, upper_bound.shape))

        # perform the forward pass for each custom layer (skipping the infinity norm layer)
        for i in range(1, len(self.layers)):
            
            # get the current layer
            l = self.layers[i]

            if VERBOSE:
                print("DeepPolyNetwork: Forward pass for layer %s of type %s, out of %s layers" % (i + 1, type(l), len(self.layers)))

            # perform the forward pass for the current layer
            x, lower_bound, upper_bound = l(x, lower_bound, upper_bound)
            
            if VERBOSE:
                print("DeepPolyNetwork forward: shape after layer %s: x: %s, lower bound %s, upper bound %s" % (i + 1, x.shape, lower_bound.shape, upper_bound.shape))
            assert x.shape == lower_bound.shape == upper_bound.shape, "DeepPolyNetwork forward: input shape mismatch after forward pass for layer %s" % (i)
            
            # if l is a linear layer, perform backsubstitution
            if type(l) == DeepPolyLinearLayer and i > 1:
            # TODO: why it doesn't work for also the linear layer?
            # if i > 1:
                if VERBOSE:
                    print("DeepPolyNetwork: Performing backsubstitution")
                
                lower_bound_tmp, upper_bound_tmp = self.backsubstitution(i, x.shape[1])

                assert lower_bound_tmp.shape == lower_bound.shape
                assert upper_bound_tmp.shape == upper_bound.shape

                # get the tightest bound
                lower_bound = torch.maximum(lower_bound, lower_bound_tmp)
                upper_bound = torch.minimum(upper_bound, upper_bound_tmp)

            self.lower_bounds_list.append(lower_bound)
            self.upper_bounds_list.append(upper_bound)
            self.activation_list.append(x)

        if VERBOSE:
            print("DeepPolyNetwork: Forward pass completed")

        assert len(self.lower_bounds_list) == len(self.upper_bounds_list), "DeepPolyNetwork forward pass completed: Error: number of lower bounds != number of upper bounds"

        return x, self.lower_bounds_list, self.upper_bounds_list
