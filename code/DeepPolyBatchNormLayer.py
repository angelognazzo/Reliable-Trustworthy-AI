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
        
        
    def forward(self, x, lower_bound, upper_bound, gamma, beta, mean, var, eps, input_shape):
        
        x = x.reshape(input_shape)
        lower_bound = lower_bound.reshape(input_shape)
        upper_bound = upper_bound.reshape(input_shape)
        
        if VERBOSE:
            print("DeepPolyBatchNormLayer: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))
        
        var_sqrt = torch.sqrt(var + eps)

        w = torch.div(gamma,var_sqrt)
        b = (- mean * gamma) / var_sqrt + beta

        self.weights = torch.diag(w)
        self.bias = b.reshape(1, -1)
        # perform batch normalization on the actual input
        x= self.layer(x)
        lower_bound = self.layer(lower_bound)
        upper_bound = self.layer(upper_bound)

        return x, lower_bound, upper_bound, input_shape
    
    

