import torch
import InfinityNormLayer
import DeepPolyLinearLayer
import DeepPolyReluLayer
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
            if type(l) == torch.nn.modules.linear.Linear:
                self.layers.append(DeepPolyLinearLayer(net, l))
            elif type(l) == torch.nn.modules.activation.ReLU:
                self.layers.append(DeepPolyReluLayer(...))
            else:
                print("DeepPolyNetwork constructor ERROR: layer type not supported")

        if VERBOSE:
            print("DeepPolyNetwork: Created %s layers (Infinity norm layer included)", len(self.layers))

        assert len(self.layers) > 0, "DeepPolyNetwork constructor: no layers created"
        assert len(self.layers) == len(self.net.layers) - 1, "DeepPolyNetwork constructor: number of layers mismatch compared to the original network"

        self.lower_bounds_list = []
        self.upper_bounds_list = []
        self.activation_list = []

    # perform backsubstitution to compute tighter lower and upper bounds
    def backsubstitution(self, input_size):

        # get the current lower and upper bounds that we want to tighten
        current_lower_bound = self.lower_bounds_list[-1]
        current_upper_bound = self.upper_bounds_list[-1]

        # initialize new lower and upper weights
        lower_weights = torch.eye(input_size)
        lower_bias = torch.zeros(input_size)
        upper_weights = torch.eye(input_size)
        upper_bias = torch.zeros(input_size)

        # iterate through the layers in reverse order starting from the second to last layer (penultimum) to and excluding the first layer (InfinityNormLayer)
        for i in range(len(self.layers) - 2, 0, -1):

            if VERBOSE:
                print("Backsubstitution: layer %s out of %s layers", i, len(self.layers))

            # get the current layer and its type
            layer = self.layers[i]
            layer_type = type(layer) == DeepPolyReluLayer

            # TODO: add to the DeepPolyRelulayer "layer.upper_weights" and "layer.lower_weights"
            # if a linear layer is encountered get the actual weights and bias of the layer, else use the computed bounds
            upper_weights_tmp = layer.upper_weights if layer_type else layer.weights
            upper_bias_tmp = layer.upper_bias if layer_type else layer.bias
            lower_weights_tmp = layer.lower_weights if layer_type else layer.weights
            lower_bias_tmp = layer.lower_bias if layer_type else layer.bias

            upper_bias += torch.matmul(upper_weights, upper_bias_tmp)
            lower_bias += torch.matmul(lower_weights, lower_bias_tmp)
            upper_weights = torch.matmul(upper_weights, upper_weights_tmp)
            lower_weights = torch.matmul(lower_weights, lower_weights_tmp)

        # perform a new forward pass with the new weights to compute the new lower and upper bounds
        new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(lower_weights, current_lower_bound, current_upper_bound, lower_bias)
        _, new_upper_bound = DeepPolyLinearLayer(upper_weights, current_lower_bound, current_upper_bound, upper_bias)

        assert new_lower_bound.shape == new_upper_bound.shape
        assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"

        return new_lower_bound, new_upper_bound

    def forward(self, x):
        # normalize and flatten the input image
        x = self.net.layers[0](x)
        x = self.net.layers[1](x)
        self.activation_list.append(x)

        # input should be flattened
        assert x.shape[0] == 1

        # perturb the input image passing the input through the infinity norm layer
        lower_bound, upper_bound = self.layers[0](x)
        # save the initial lower and upper bounds
        self.lower_bounds_list.append(lower_bound)
        self.upper_bounds_list.append(upper_bound)

        # perform the forward pass for each custom layer (skipping the infinity norm layer)
        for i, l in enumerate(self.layers(1, len(self.layers))):

            if VERBOSE:
                print("DeepPolyNetwork: Forward pass for layer %s, out of %s layers", i, len(self.layers))
                print("DeepPolyNetwork: forward pass for layer of type %s", type(l))

            # perform the forward pass for the current layer
            x, lower_bound, upper_bound = l(x, lower_bound, upper_bound)

            # if l is a linear layer, perform backsubstitution
            if type(l) == DeepPolyLinearLayer:
                if VERBOSE:
                    print("DeepPolyNetwork: Performing backsubstitution")

                lower_bound_tmp, upper_bound_tmp = self.backsubstitution(x.shape[0])

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
