import torch
from settings import VERBOSE
from backsubstitution import backsubstitution

class DeepPolyLinearLayer(torch.nn.Module):
    """
    Class implementing the LinearLayer of the DeepPoly algorithm
    """

    def __init__(self, layer, previous_layers) -> None:
        super().__init__()
        # self.layer = layer
        self.previous_layers = previous_layers
        # we want 784x50 and not the opposite
        self.lower_weights = layer.weight.detach().t()
        self.upper_weights = layer.weight.detach().t()
        # we want 1x50 and not the opposite
        self.lower_bias = layer.bias.detach().reshape(1, -1)
        self.upper_bias = layer.bias.detach().reshape(1, -1)
        
        self.isRes=False
        
    # forward pass through the network
    def forward(self, lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag):
        if flag==True:
            lower_bound, upper_bound, _, _, _, _= backsubstitution(self.previous_layers + [self], first_lower_bound, first_upper_bound)
        else:
            bounds_upper=torch.empty_like(self.upper_bias)
            bounds_lower=torch.empty_like(self.lower_bias)
            upper_bound=bounds_upper.fill_( float("Inf"))
            lower_bound=bounds_lower.fill_( -float("Inf"))

        if VERBOSE:
            print("DeepPolyLinearLayer: lower_boud shape %s" %(str(lower_bound.shape)))
        
                
        return lower_bound, upper_bound
