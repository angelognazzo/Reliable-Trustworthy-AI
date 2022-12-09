import torch
from settings import VERBOSE
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from InfinityNormLayer import InfinityNormLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer

"""
This file contains all the functions needed to perform backsubstitution
"""
# helper function of handle_backsubstitution_resnet_block
def compute_new_weights_and_bias(layers, starting_lower_weights, starting_upper_weights, starting_lower_bias, starting_upper_bias):

    lower_bias = torch.zeros(starting_lower_bias.shape)
    upper_bias = torch.zeros(starting_upper_bias.shape)
    lower_weights = torch.clone(starting_lower_weights)
    upper_weights = torch.clone(starting_upper_weights)

    for i in range(len(layers) - 1, -1, -1):
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

    return lower_weights, upper_weights, lower_bias, upper_bias

# handle the backsubstitution for a resnet block
# find the new weight matrix multiplying the weight matrices of the two path separately
def handle_backsubstitution_resnet_block(resnet_block, lower_weights, upper_weights, lower_bias, upper_bias):

    lower_weights_a, upper_weights_a, lower_bias_a, upper_bias_a = compute_new_weights_and_bias(
        resnet_block.path_a, lower_weights, upper_weights, lower_bias, upper_bias)
    lower_weights_b, upper_weights_b, lower_bias_b, upper_bias_b = compute_new_weights_and_bias(
        resnet_block.path_b, lower_weights, upper_weights, lower_bias, upper_bias)

    lower_weights = lower_weights_a + lower_weights_b
    upper_weights = upper_weights_a + upper_weights_b
    lower_bias = lower_bias_a + lower_bias_b + lower_bias
    upper_bias = upper_bias_a + upper_bias_b + upper_bias

    return lower_weights, upper_weights, lower_bias, upper_bias

# perform backsubstitution to compute tighter lower and upper bounds
def backsubstitution(layers, current_layer, input_size, first_lower_bound, first_upper_bound):    
    # initialize new lower and upper weights
    lower_weights = torch.eye(input_size)
    lower_bias = torch.zeros(1, input_size)
    upper_weights = torch.eye(input_size)
    upper_bias = torch.zeros(1, input_size)

    # iterate through the layers in reverse order starting from the layer before the current_layer to the beginning of the network
    for i in range(current_layer, -1, -1):

        if VERBOSE:
            print("Backsubstitution Loop: layer %s out of %s layers" % (i, current_layer))
            
        # get the current layer and its type
        layer = layers[i]
        
        if type(layer) == InfinityNormLayer:
            continue
        
        # check if the current layer is a resnet block (cannot do type(layer) == ResnetBlock because of circular import)
        isDeepPolyResenetBlock = not (type(layer) == DeepPolyLinearLayer or type(
            layer) == DeepPolyConvolutionalLayer or type(layer) == DeepPolyIdentityLayer or type(layer) == DeepPolyReluLayer)
        
        # if a linear layer or convolutional is encountered get the actual weights and bias of the layer,
        # if a RELU layer is encountered use the computed weight bounds
        # if a resnet block is encountered, skip-it, this case will be handled by the if statement below, 
        # because if the previous layer is a resnet block, then we need to be careful abous the bias, in the sense that we don't want
        # to add it twice
        if isDeepPolyResenetBlock:
            continue 
        elif type(layer) == DeepPolyReluLayer:
            upper_weights_tmp = layer.upper_weights
            upper_bias_tmp = layer.upper_bias
            lower_weights_tmp = layer.lower_weights
            lower_bias_tmp = layer.lower_bias
        else:
            upper_weights_tmp = layer.weights
            upper_bias_tmp = layer.bias
            lower_weights_tmp = layer.weights
            lower_bias_tmp = layer.bias


        upper_bias += torch.matmul(upper_bias_tmp, upper_weights)
        lower_bias += torch.matmul(lower_bias_tmp, lower_weights)
        upper_weights = torch.matmul(upper_weights_tmp, upper_weights)
        lower_weights = torch.matmul(lower_weights_tmp, lower_weights)
        
        # if the previous layer is a resnet block we handle this case manually bacause we need to be careful about the bias
        isDeepPolyResenetBlock = not (type(layers[i-1]) == DeepPolyLinearLayer or type(layers[i-1]) == DeepPolyConvolutionalLayer or type(
            layers[i-1]) == DeepPolyIdentityLayer or type(layers[i-1]) == DeepPolyReluLayer or type(layers[i-1]) == InfinityNormLayer)
        if i > 0 and isDeepPolyResenetBlock:
            lower_weights, upper_weights, lower_bias, upper_bias = handle_backsubstitution_resnet_block(
                layers[i-1], lower_weights, upper_weights, lower_bias, upper_bias)

    # perform a new forward pass with the new weights to compute the new lower and upper bounds
    new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, lower_weights, lower_bias)
    _, new_upper_bound = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, upper_weights, upper_bias)

    assert new_lower_bound.shape == new_upper_bound.shape
    # assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"

    return new_lower_bound, new_upper_bound
