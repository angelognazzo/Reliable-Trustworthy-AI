
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
        self.kernel = layer.weight
        self.weights = None 
        
        # The bias is a 1D tensor, we want to reshape it to a 2D tensor??? TODO: check this
        self.bias_kernel = layer.bias
        self.bias = None
        
        self.stride = layer.stride
        self.padding = layer.padding
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer: weights shape: %s, bias shape %s, stride %s, padding %s" % (
                str(self.kernel.shape), str(self.bias_kernel.shape), str(self.stride), str(self.padding)))

    # swap the bounds depending on the sign of the weights
    # return new lower and upper bounds
    def swap_and_forward(self, lower_bound, upper_bound, weights, bias, stride, padding):
        negative_mask = (weights < 0).int()
        positive_mask = (weights >= 0).int()

        negative_weights = torch.mul(negative_mask, weights)
        positive_weights = torch.mul(positive_mask, weights)
 
        lower_bound_new = F.conv2d(upper_bound, negative_weights, bias, stride, padding) + \
            F.conv2d(lower_bound, positive_weights,bias, stride, padding) + bias.reshape(1, -1, 1, 1)

        upper_bound_new = F.conv2d(lower_bound,  negative_weights, bias, stride, padding) + \
            F.conv2d(upper_bound, positive_weights, bias, stride, padding) + bias.reshape(1, -1, 1, 1)
        
        return lower_bound_new.flatten(start_dim=1, end_dim=-1), upper_bound_new.flatten(start_dim=1, end_dim=-1)
    
    
    def forward(self, x, lower_bound, upper_bound, input_shape):
        # x, lower_bound and upper_bound are flattened (i.e. [1, 3072]), we want to reshape them to being a tensor so that we can perfrom the convolutions(i.e [1, 3, 32, 32])
        x = x.reshape(input_shape)
        lower_bound = lower_bound.reshape(input_shape)
        upper_bound = upper_bound.reshape(input_shape)
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer RESHAPE: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))
        
        lower_bound, upper_bound = self.swap_and_forward(
            lower_bound, upper_bound, self.kernel, self.bias_kernel, self.stride, self.padding)
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))
        
        x = self.layer(x)
        
        #################################################################
        #################################################################
        #################################################################
        w = torch.eye(input_shape.numel()).view(list(input_shape) + [input_shape.numel()])
        w = w.permute(0, 1, 4, 2, 3)

        w = F.conv3d(w, self.kernel.unsqueeze(2), stride=tuple(
            [1] + list(self.stride)), padding=tuple([0] + list(self.padding))).permute(0, 1, 3, 4, 2)
        # remove the first empty dimension
        w = w[0]

        self.weights = torch.flatten(w, start_dim=0, end_dim=2).t()

        b = torch.ones(w.shape[:-1]) * self.bias_kernel[:, None, None]
        self.bias = torch.flatten(b)

        #################################################################
        #################################################################
        #################################################################
        
        input_shape = x.shape
        x = x.flatten(start_dim=1, end_dim=-1)

        assert lower_bound.shape == x.shape == upper_bound.shape, "swap_and_forward CNN: lower and upper bounds have different shapes"
        assert (lower_bound <= upper_bound).all(), "swap_and_forward CNN: error with the box bounds: lower > upper"
          
        return x, lower_bound, upper_bound, input_shape