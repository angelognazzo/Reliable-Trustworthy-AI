import torch
from InfinityNormLayer import InfinityNormLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyResnetBlock import DeepPolyResnetBlock
from settings import VERBOSE
import networks
import resnet


class DeepPolyNetwork(torch.nn.Module):
    """
    Perform a DeepPoly analysis on the given network
    """
    def __init__(self, net, eps) -> None:
        super().__init__()

        self.eps = eps
        self.net = net
        
        # contains the custom layers of the network
        self.layers = [InfinityNormLayer(self.eps)]
        
        # parse the network to understand which layers to create
        # in case of a resnet, parsing is more complicated because layers are nested in blocks
        layers_to_create = []
        if type(self.net == networks.NormalizedResnet):
            
            # get the correct normalization layer
            self.normalization_layer = self.net.normalization
            #  get the actual resnet from the 'normalized resnet' object
            self.net = self.net.resnet
            
            # get all the layers of the resnet
            l = list(self.net.modules())
            i = 1
            while i < len(l):
                
                module = l[i]
                # we don't care about sequentials blocks. go to the next layer
                if type(module) == torch.nn.modules.Sequential:
                    i += 1
                    continue
                # append the current layer
                layers_to_create.append(module)
                
                # if the current layer is of type 'BasicBlock' I want to get the layer as a whole and skip
                # all the layers inside the block
                if type(module) == resnet.BasicBlock:
                    i += len(module.path_a) + len(module.path_b) + 2 + 1
                else:
                    i += 1
        # in case of a normal network, parsing is easier, just get the correct field
        else:
            layers_to_create = self.net.layers
            self.normalization_layer = layers_to_create[0]
       
        # create a custom layer for each layer in the original network
        for i in range(0, len(layers_to_create)):
            l = layers_to_create[i]
            
            # skip the flattening layer and the normalization.
            # These layers are not present in every network, For example in CNNs and ResNets. That's why this check is necessary
            if type(l) == torch.nn.modules.flatten.Flatten or type(l) == torch.nn.modules.normalization.LayerNorm:
                continue
            
            # create a custom layer based on the type of the original layer
            if type(l) == torch.nn.modules.linear.Linear:
                self.layers.append(DeepPolyLinearLayer(net, l))
            elif type(l) == torch.nn.modules.activation.ReLU:
                self.layers.append(DeepPolyReluLayer(l))
            elif type(l) == torch.nn.modules.Conv2d:
                self.layers.append(DeepPolyConvolutionalLayer(l))
            elif type(l) == resnet.BasicBlock:
                self.layers.append(DeepPolyResnetBlock(l))
            else:
                print("DeepPolyNetwork constructor ERROR: layer type not supported")

        if VERBOSE:
            print("DeepPolyNetwork: Created %s layers (Infinity norm layer included)" % (len(self.layers)))

        assert len(self.layers) > 0, "DeepPolyNetwork constructor: no layers created"

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
        # iterate through the layers in reverse order starting from the layer before the current_layer to and excluding the first layer (InfinityNormLayer)
        for i in range(current_layer, 0, -1):

            if VERBOSE:
                print("Backsubstitution Loop: layer %s out of %s layers" % (i, current_layer))
                
            # get the current layer and its type
            layer = self.layers[i]
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
        new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, lower_weights, lower_bias)
        _, new_upper_bound = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, upper_weights, upper_bias)

        assert new_lower_bound.shape == new_upper_bound.shape
        # assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"

        return new_lower_bound, new_upper_bound

    def forward(self, x):
        # perturb the input image passing the input through the infinity norm custom layer
        lower_bound, upper_bound = self.layers[0](x)
        input_shape = x.shape
        
        # normalize the input image and flatten
        lower_bound = self.normalization_layer(lower_bound).flatten().reshape(1, -1)
        upper_bound = self.normalization_layer(upper_bound).flatten().reshape(1, -1)

        # save the initial lower and upper bounds
        self.lower_bounds_list.append(lower_bound)
        self.upper_bounds_list.append(upper_bound)
       
        # pass x through normalization layers and flatten
        x = self.normalization_layer[0](x)
        x = x.flatten().reshape(1, -1)
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
            
            # if l not a RELU layer and we are at least at the second layer: perform backsubstitution
            if (type(l) == DeepPolyLinearLayer or type(l) == DeepPolyConvolutionalLayer) and i > 1:
                if VERBOSE:
                    print("DeepPolyNetwork: Performing backsubstitution")
                
                # perform backsubstitution
                lower_bound_tmp, upper_bound_tmp = self.backsubstitution(i, x.shape[1])
                
                # dimensions should be preserved after backsubstitution
                assert lower_bound_tmp.shape == lower_bound.shape
                assert upper_bound_tmp.shape == upper_bound.shape

                # get the tightest bounds possible
                # lower_bound and lower_bound_tmp could or couldn't have an interesection
                # EASY CASE: there is an interesection between the two bounds:
                # check the intersection condition
                mask_positive = torch.max(lower_bound_tmp, lower_bound) <= torch.min(upper_bound_tmp, upper_bound)
                mask_negative = torch.logical_not(mask_positive)
                # if there is an intersection, the bounds can be tighten by taking the greatest maximum and the smallest minimum 
                lower_bound = torch.where(mask_positive, torch.max(lower_bound_tmp, lower_bound), lower_bound)
                upper_bound =  torch.where(mask_positive, torch.min(upper_bound_tmp, upper_bound), upper_bound)
                
                assert (lower_bound <= upper_bound).all(), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"
                
                # there is no intersection between the two bounds, bounds are therefore 'disjoint'
                # check if the new bounds are tighter than the old ones and if upper_bound_tmp > lower_bound_tmp.
                # This (upper_bound_tmp > lower_bound_tmp) could happen during backsubstitution where upper_bound_tmp
                # and lower_bound_tmp are computed using two different calls to the 'swap_and_forward' function 
                # and therefore it's possible that in some cases the correct order is not preserved
                mask_tighter_disjoint = (upper_bound_tmp - lower_bound_tmp < upper_bound - lower_bound) & (upper_bound_tmp - lower_bound_tmp >= 0)
                # use also mask_negative to update the entries that haven't been updated above
                lower_bound = torch.where(mask_negative & mask_tighter_disjoint, lower_bound_tmp, lower_bound)
                upper_bound = torch.where(mask_negative & mask_tighter_disjoint, upper_bound_tmp, upper_bound)
                
                # the correct order now should be respected
                assert (lower_bound <= upper_bound).all(), "DeepPolyNetwork forward: Error with the box bounds: lower > upper"
                
            # save the newly computed bounds
            self.lower_bounds_list.append(lower_bound)
            self.upper_bounds_list.append(upper_bound)
            self.activation_list.append(x)

        if VERBOSE:
            print("DeepPolyNetwork: Forward pass completed")

        # number of lower and upper bounds should be the same
        assert len(self.lower_bounds_list) == len(self.upper_bounds_list), "DeepPolyNetwork forward pass completed: Error: number of lower bounds != number of upper bounds"
        # the correct order now should be respected
        assert (lower_bound <= upper_bound).all(), "DeepPolyNetwork forward pass completed: Error with the box bounds: lower > upper"
        
        return x, self.lower_bounds_list, self.upper_bounds_list
