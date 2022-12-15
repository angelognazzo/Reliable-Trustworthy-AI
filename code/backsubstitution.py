import torch
from settings import VERBOSE
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from InfinityNormLayer import InfinityNormLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyIdentityLayer import DeepPolyIdentityLayer
from DeepPolyBatchNormLayer import DeepPolyBatchNormLayer
"""
This file contains all the functions needed to perform backsubstitution
"""
# helper function of handle_backsubstitution_resnet_block
def compute_new_weights_and_bias(layers, starting_lower_weights, starting_upper_weights, starting_lower_bias, starting_upper_bias):

    FLAG = True
    
    lower_bias = torch.zeros(starting_lower_bias.shape)
    upper_bias = torch.zeros(starting_upper_bias.shape)
    lower_weights = torch.clone(starting_lower_weights)
    upper_weights = torch.clone(starting_upper_weights)
    
    for i in range(len(layers) - 1, -1, -1): #range(len(layers) - 1, -1, -1)
        
        layer = layers[i]
        
        isDeepPolyResenetBlock = not (type(layer) == DeepPolyLinearLayer or type(
            layer) == DeepPolyConvolutionalLayer or type(layer) == DeepPolyIdentityLayer or type(layer) == DeepPolyReluLayer
            or type(layer) == DeepPolyBatchNormLayer)
        if isDeepPolyResenetBlock:
            FLAG = False
            lower_weights_a = torch.clone(lower_weights)#torch.eye(max(lower_bias.shape[0], lower_bias.shape[1]))
            lower_bias_a = torch.zeros_like(lower_bias)
            upper_weights_a = torch.clone(upper_weights)#torch.eye(max(upper_bias.shape[0], upper_bias.shape[1]))
            upper_bias_a = torch.zeros_like(upper_bias)
            
            for i in range(len(layer.path_a) - 1, -1, -1):
                layer_inside = layer.path_a[i]
                layer_type_inside = type(layer_inside) == DeepPolyReluLayer
                # if a linear layer or convolutional is encountered get the actual weights and bias of the layer,
                # else (RELU layer) use the computed weight bounds
                upper_weights_tmp_a = layer_inside.upper_weights if layer_type_inside else layer_inside.weights
                upper_bias_tmp_a = layer_inside.upper_bias if layer_type_inside else layer_inside.bias
                lower_weights_tmp_a = layer_inside.lower_weights if layer_type_inside else layer_inside.weights
                lower_bias_tmp_a = layer_inside.lower_bias if layer_type_inside else layer_inside.bias
                
                upper_bias_a += torch.matmul(upper_bias_tmp_a, upper_weights_a)
                lower_bias_a += torch.matmul(lower_bias_tmp_a, lower_weights_a)
                upper_weights_a = torch.matmul(upper_weights_tmp_a, upper_weights_a)
                lower_weights_a = torch.matmul(lower_weights_tmp_a, lower_weights_a)
            
                # torch.eye(max(lower_bias.shape[0], lower_bias.shape[1]))
            lower_weights_b = torch.clone(lower_weights)
            lower_bias_b = torch.zeros_like(lower_bias)
            # torch.eye(max(upper_bias.shape[0], upper_bias.shape[1]))
            upper_weights_b = torch.clone(upper_weights)
            upper_bias_b = torch.zeros_like(upper_bias)
            for i in range(len(layer.path_b) - 1, -1, -1):
                layer_inside = layer.path_b[i]
                layer_type_inside = type(layer_inside) == DeepPolyReluLayer
                # if a linear layer or convolutional is encountered get the actual weights and bias of the layer,
                # else (RELU layer) use the computed weight bounds
                upper_weights_tmp_b = layer_inside.upper_weights if layer_type_inside else layer_inside.weights
                upper_bias_tmp_b = layer_inside.upper_bias if layer_type_inside else layer_inside.bias
                lower_weights_tmp_b = layer_inside.lower_weights if layer_type_inside else layer_inside.weights
                lower_bias_tmp_b = layer_inside.lower_bias if layer_type_inside else layer_inside.bias

                upper_bias_b += torch.matmul(upper_bias_tmp_b, upper_weights_b)
                lower_bias_b += torch.matmul(lower_bias_tmp_b, lower_weights_b)
                upper_weights_b = torch.matmul(
                    upper_weights_tmp_b, upper_weights_b)
                lower_weights_b = torch.matmul(
                    lower_weights_tmp_b, lower_weights_b)

            # manca for b
            res_lower_weights=lower_weights_a +lower_weights_b
            res_upper_weights=upper_weights_a +upper_weights_b
            res_lower_bias=lower_bias_a + lower_bias_b
            res_upper_bias=upper_bias_a + upper_bias_b
            # print("problema qui con quello che esce di bias da identity layer")
            
            upper_weights_tmp=res_upper_weights
            upper_bias_tmp=res_upper_bias# torch.squeeze(res_upper_bias)
            lower_weights_tmp=res_lower_weights
            lower_bias_tmp=res_lower_bias#torch.squeeze(res_lower_bias)
        elif type(layer) == DeepPolyReluLayer:
            FLAG = True
            upper_weights_tmp = layer.upper_weights
            upper_bias_tmp = layer.upper_bias#torch.squeeze(layer.upper_bias)
            lower_weights_tmp = layer.lower_weights
            lower_bias_tmp = layer.lower_bias#torch.squeeze(layer.lower_bias)
            
        else:
            FLAG = True
            upper_weights_tmp = layer.weights
            upper_bias_tmp = layer.bias#torch.squeeze(layer.bias)
            lower_weights_tmp = layer.weights
            lower_bias_tmp = layer.bias#torch.squeeze(layer.bias)

        # why when i is = 0 this thing is printed two times???? 
        if FLAG:
            upper_bias += torch.matmul(upper_bias_tmp, upper_weights)
            lower_bias += torch.matmul(lower_bias_tmp, lower_weights)
            upper_weights = torch.matmul(upper_weights_tmp, upper_weights)
            lower_weights = torch.matmul(lower_weights_tmp, lower_weights)
        else:
            upper_weights = upper_weights_tmp
            upper_bias = upper_bias_tmp
            lower_weights = lower_weights_tmp
            lower_bias = lower_bias_tmp
            
        
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
            layer) == DeepPolyConvolutionalLayer or type(layer) == DeepPolyIdentityLayer or type(layer) == DeepPolyReluLayer
                                      or type(layer) == DeepPolyBatchNormLayer)
        
        # if a linear layer or convolutional is encountered get the actual weights and bias of the layer,
        # if a RELU layer is encountered use the computed weight bounds
        # if a resnet block is encountered, skip-it, this case will be handled by the if statement below, 
        # because if the previous layer is a resnet block, then we need to be careful abous the bias, in the sense that we don't want
        # to add it twice
        if isDeepPolyResenetBlock and i == current_layer:
            lower_weights, upper_weights, lower_bias, upper_bias = handle_backsubstitution_resnet_block(
                layer, lower_weights, upper_weights, lower_bias, upper_bias)
        elif isDeepPolyResenetBlock and i < current_layer:
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

        if not (isDeepPolyResenetBlock and i == current_layer):
            upper_bias += torch.matmul(upper_bias_tmp, upper_weights)
            lower_bias += torch.matmul(lower_bias_tmp, lower_weights)
            upper_weights = torch.matmul(upper_weights_tmp, upper_weights)
            lower_weights = torch.matmul(lower_weights_tmp, lower_weights)
        
        # if the previous layer is a resnet block we handle this case manually bacause we need to be careful about the bias
        isDeepPolyResenetBlock = not (type(layers[i-1]) == DeepPolyLinearLayer or type(layers[i-1]) == DeepPolyConvolutionalLayer or type(
            layers[i-1]) == DeepPolyIdentityLayer or type(layers[i-1]) == DeepPolyReluLayer or type(layers[i-1]) == InfinityNormLayer
            or type(layers[i-1]) == DeepPolyBatchNormLayer)
        if i > 0 and isDeepPolyResenetBlock:
            lower_weights, upper_weights, lower_bias, upper_bias = handle_backsubstitution_resnet_block(
                layers[i-1], lower_weights, upper_weights, lower_bias, upper_bias)

    # perform a new forward pass with the new weights to compute the new lower and upper bounds
    new_lower_bound, _ = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, lower_weights, lower_bias)
    _, new_upper_bound = DeepPolyLinearLayer.swap_and_forward(first_lower_bound, first_upper_bound, upper_weights, upper_bias)

    assert new_lower_bound.shape == new_upper_bound.shape
    # assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"

    return new_lower_bound, new_upper_bound
