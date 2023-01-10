import torch
from settings import VERBOSE
"""
This file contains all the functions needed to perform backsubstitution
"""

def swap_and_forward_matrix(lower_weights_cumulative, upper_weights_cumulative, lower_bias_cumulative, upper_bias_cumulative, lower_weights_current_layer, upper_weights_current_layer, lower_bias_current_layer, upper_bias_current_layer):

    pivot_matrix = torch.zeros_like(lower_weights_cumulative)
    
    lower_weights_positive = torch.max(lower_weights_cumulative, pivot_matrix)
    lower_weights_negative = torch.min(lower_weights_cumulative, pivot_matrix)

    lower_weights_cumulative = lower_weights_current_layer @ lower_weights_positive + upper_weights_current_layer @ lower_weights_negative
    lower_bias_cumulative = lower_bias_current_layer @ lower_weights_positive + upper_bias_current_layer @ lower_weights_negative + lower_bias_cumulative

    upper_weights_positive = torch.max(upper_weights_cumulative, pivot_matrix)
    upper_weights_negative = torch.min(upper_weights_cumulative, pivot_matrix)

    upper_weights_cumulative = upper_weights_current_layer @ upper_weights_positive + lower_weights_current_layer @ upper_weights_negative
    upper_bias_cumulative = upper_bias_current_layer @ upper_weights_positive + lower_bias_current_layer @ upper_weights_negative + upper_bias_cumulative

    return lower_weights_cumulative, upper_weights_cumulative, lower_bias_cumulative, upper_bias_cumulative

def backsubstitution(layers, first_lower_bound, first_upper_bound):
    lower_weights_cumulative = torch.clone(layers[-1].lower_weights)
    upper_weights_cumulative = torch.clone(layers[-1].upper_weights)
    lower_bias_cumulative = torch.clone(layers[-1].lower_bias)
    upper_bias_cumulative = torch.clone(layers[-1].upper_bias)
    for i in range(len(layers)-2,-1, -1):
        layer = layers[i]
        if layer.isRes==True:
            lower_bias_to_sum=torch.clone(lower_bias_cumulative)
            upper_bias_to_sum=torch.clone(upper_bias_cumulative)
            
            lower_weights_cumulative_a=torch.clone(lower_weights_cumulative)
            lower_bias_cumulative_a=torch.zeros_like(lower_bias_cumulative) 
            upper_weights_cumulative_a=torch.clone(upper_weights_cumulative)
            upper_bias_cumulative_a=torch.zeros_like(upper_bias_cumulative) 

            for j in range(len(layer.path_a)-1,-1,-1):
                layer_path_a = layer.path_a[j]
                lower_weights_cumulative_a, upper_weights_cumulative_a, lower_bias_cumulative_a, upper_bias_cumulative_a = swap_and_forward_matrix(
                    lower_weights_cumulative_a, upper_weights_cumulative_a, lower_bias_cumulative_a, upper_bias_cumulative_a, layer_path_a.lower_weights, layer_path_a.upper_weights, layer_path_a.lower_bias, layer_path_a.upper_bias)

            lower_weights_cumulative_b=torch.clone(lower_weights_cumulative)
            lower_bias_cumulative_b=torch.zeros_like(lower_bias_cumulative) 
            upper_weights_cumulative_b=torch.clone(upper_weights_cumulative)
            upper_bias_cumulative_b=torch.zeros_like(upper_bias_cumulative) 

            for j in range(len(layer.path_b)-1,-1,-1):
                layer_path_b = layer.path_b[j]
                lower_weights_cumulative_b, upper_weights_cumulative_b, lower_bias_cumulative_b, upper_bias_cumulative_b = swap_and_forward_matrix(
                    lower_weights_cumulative_b, upper_weights_cumulative_b, lower_bias_cumulative_b, upper_bias_cumulative_b, layer_path_b.lower_weights, layer_path_b.upper_weights, layer_path_b.lower_bias, layer_path_b.upper_bias)
            
            lower_weights_cumulative=lower_weights_cumulative_a+lower_weights_cumulative_b
            upper_weights_cumulative=upper_weights_cumulative_a+upper_weights_cumulative_b
            upper_bias_cumulative=upper_bias_cumulative_a+upper_bias_cumulative_b+upper_bias_to_sum
            lower_bias_cumulative=lower_bias_cumulative_a+lower_bias_cumulative_b+lower_bias_to_sum
            i=i-1
        else: 
            lower_weights_cumulative, upper_weights_cumulative, lower_bias_cumulative, upper_bias_cumulative = swap_and_forward_matrix(
                    lower_weights_cumulative, upper_weights_cumulative, lower_bias_cumulative, upper_bias_cumulative, layer.lower_weights, layer.upper_weights, layer.lower_bias, layer.upper_bias)

    pivot_matrix = torch.zeros_like(lower_weights_cumulative)
    new_lower_bound = first_lower_bound @ torch.max(lower_weights_cumulative, pivot_matrix) +  first_upper_bound @ torch.min(lower_weights_cumulative, pivot_matrix) + lower_bias_cumulative
    new_upper_bound = first_upper_bound @ torch.max(upper_weights_cumulative, pivot_matrix) + first_lower_bound @ torch.min(upper_weights_cumulative, pivot_matrix) +  upper_bias_cumulative
    
    #assert new_lower_bound.shape == new_upper_bound.shape
    #assert (new_lower_bound <= new_upper_bound).all(), "Backsubstitution: Error with the box bounds: lower > upper"    

    return new_lower_bound, new_upper_bound, lower_weights_cumulative, upper_weights_cumulative, lower_bias_cumulative, upper_bias_cumulative
                        