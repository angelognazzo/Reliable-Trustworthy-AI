import torch
from settings import VERBOSE


class DeepPolyBatchNormLayer(torch.nn.Module):
    """
    Class implementing the BatchNorm2d layer of the DeepPoly algorithm
    """
    
    def __init__(self, layer, previous_layers, input_shape):
        super().__init__()
        
        self.layer = layer
        self.gamma = layer.weight
        self.beta=layer.bias
        self.mean=layer.running_mean
        self.var=layer.running_var
        self.eps=layer.eps
        self.input_shape = (1, input_shape[0], input_shape[1], input_shape[2])
        self.input_shape_flatten = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None
        
        self.isRes=False
        
    def forward(self, lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag=False):
        
        # compute the weights and bias of the layer
        shape = self.input_shape
        
        var = self.var.reshape(-1, 1)  # (C, 1)
        var = var.repeat(1, shape[2] * shape[3])  # (C, H*W)
        var = var.flatten()  # (C*H*W, )
        var = torch.sqrt(var + self.eps)  # sqrt(var_i + eps)
        var = 1 / var
        gamma = self.gamma.reshape(-1, 1)
        gamma = gamma.repeat(1, shape[2] * shape[3])
        gamma = gamma.flatten()

        weights = torch.diag(gamma * var)
        self.lower_weights = weights
        self.upper_weights = weights

        mean = self.mean.reshape(-1, 1)  # (C, 1)
        mean = mean.repeat(1, shape[2] * shape[3])  # (C, H*W)
        mean = mean.flatten()  # (C*H*W, )
        bias = mean * var * gamma
        bias_final = bias + self.beta.reshape(-1, 1).repeat(1, shape[2] * shape[3]).flatten().reshape(1, -1)
        self.lower_bias = bias_final
        self.upper_bias = bias_final
       
        # perform batch normalization on the actual input
        lower_bound = self.layer(lower_bound.reshape(shape)).flatten().reshape(1, -1)
        upper_bound = self.layer(upper_bound.reshape(shape)).flatten().reshape(1, -1)
        
        if VERBOSE:
            print("DeepPolyBatchNormLayer: lower_bound shape %s, upper_bound shape %s" % (
                str(lower_bound.shape), str(upper_bound.shape)))

        return lower_bound, upper_bound
    
    

