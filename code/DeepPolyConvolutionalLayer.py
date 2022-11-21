
import torch
from settings import VERBOSE
import torch.nn.functional as F

class DeepPolyConvolutionalLayer(torch.nn.Module):
    """
    Class implementing the ConvolutionalLayer of the DeepPoly algorithm
    """
    
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.weights = layer.weight
        
        # The bias is a 1D tensor, we want to reshape it to a 2D tensor??? TODO: check this
        self.bias = layer.bias
        
        self.stride = layer.stride
        self.padding = layer.padding
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer: weights shape: %s, bias shape %s, stride %s, %s padding" % (
                str(self.weights.shape), str(self.bias.shape), str(self.stride), str(self.padding)))

    # swap the bounds depending on the sign of the weights
    # return new lower and upper bounds
    @staticmethod
    def swap_and_forward(lower_bound, upper_bound, weights, bias, stride, padding):
        negative_mask = (weights < 0).int()
        positive_mask = (weights >= 0).int()

        negative_weights = torch.mul(negative_mask, weights)
        positive_weights = torch.mul(positive_mask, weights)
 
        lower_bound_new = F.conv2d(upper_bound, negative_weights, bias, stride, padding) + \
            F.conv2d(lower_bound, positive_weights,bias, stride, padding) + bias.reshape(1, -1, 1, 1)

        upper_bound_new = F.conv2d(lower_bound,  negative_weights, bias, stride, padding) + \
            F.conv2d(upper_bound, positive_weights, bias, stride, padding) + bias.reshape(1, -1, 1, 1)
        
        return lower_bound_new, upper_bound_new
        
    def forward(self, x, lower_bound, upper_bound):
        lower_bound, upper_bound = self.swap_and_forward(
            lower_bound, upper_bound, self.weights, self.bias, self.stride, self.padding)
        
        x = self.layer(x)
        
        if VERBOSE:
            print("DeepPolyLinearLayer: x shape %s" % (str(x.shape)))

        assert lower_bound.shape == x.shape == upper_bound.shape, "swap_and_forward CNN: lower and upper bounds have different shapes"
        assert (lower_bound <= upper_bound).all(), "swap_and_forward CNN: error with the box bounds: lower > upper"
          
        return x, lower_bound, upper_bound