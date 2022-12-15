import torch
from settings import VERBOSE


class DeepPolyBatchNormLayer(torch.nn.Module):
    """
    Class implementing the BatchNorm2d layer of the DeepPoly algorithm
    """
    
    def __init__(self, layer):
        super().__init__()
        
        self.layer = layer
        self.gamma = layer.weight
        self.beta=layer.bias
        self.mean=layer.running_mean
        self.var=layer.running_var
        self.eps=layer.eps
        
        self.weights = None
        self.bias = None
        
        
    def forward(self, x, lower_bound, upper_bound, input_shape):
        
        # compute the weights and bias of the layer
        shape = input_shape
        
        var = self.var.reshape(-1, 1)  # (C, 1)
        var = var.repeat(1, shape[2] * shape[3])  # (C, H*W)
        var = var.flatten()  # (C*H*W, )
        var = torch.sqrt(var + self.eps)  # sqrt(var_i + eps)
        var = 1 / var
        gamma = self.gamma.reshape(-1, 1)
        gamma = gamma.repeat(1, shape[2] * shape[3])
        gamma = gamma.flatten()

        self.weights = torch.diag(gamma * var)

        mean = self.mean.reshape(-1, 1)  # (C, 1)
        mean = mean.repeat(1, shape[2] * shape[3])  # (C, H*W)
        mean = mean.flatten()  # (C*H*W, )
        bias = mean * var * gamma
        self.bias = bias + self.beta.reshape(-1, 1).repeat(1, shape[2] * shape[3]).flatten().reshape(1, -1)
       
        # perform batch normalization on the actual input
        x = x.reshape(input_shape)
        x = self.layer(x)
        x = x.flatten().reshape(1, -1)
        lower_bound = self.layer(lower_bound.reshape(shape)).flatten().reshape(1, -1)
        upper_bound = self.layer(upper_bound.reshape(shape)).flatten().reshape(1, -1)
        
        if VERBOSE:
            print("DeepPolyBatchNormLayer: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))

        return x, lower_bound, upper_bound, input_shape
    
    

