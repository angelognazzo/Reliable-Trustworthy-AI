import torch
from settings import VERBOSE


class DeepPolyReluLayer(torch.nn.Module):
    """
    Class implementing the ReluLayer of the DeepPoly algorithm
    """

    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None

    def forward(self, x, lower, upper):
        
        if VERBOSE:
            print("DeepPolyReluLayer forward: lower shape %s, upper shape %s" % (str(lower.shape), str(upper.shape)))
        
        assert lower.shape == upper.shape, "DeepPolyReluLayer forward: lower and upper bounds have different shapes"
        
        # compute the relu on the input
        x = self.layer(x)
        
        lower_to_return = torch.zeros_like(lower)
        upper_to_return = torch.zeros_like(upper)
        self.lower_weights = torch.zeros(lower.shape[1], lower.shape[1])
        self.upper_weights = torch.zeros(lower.shape[1], lower.shape[1])
        self.lower_bias = torch.zeros_like(lower)
        self.upper_bias = torch.zeros_like(upper)

        for i in range(lower.shape[1]):
            
            l = lower[0, i]
            u = upper[0, i]
            
            # all the points are negative
            if u <= 0:
                if VERBOSE:
                    print("DeepPolyReluLayer forward: all the points are negative")
                pass 
                
            # all the points are positive
            elif l >= 0:
                if VERBOSE:
                    print("DeepPolyReluLayer forward: all the points are positive")
                lower_to_return[0,i] = u
                upper_to_return[0,i] = u
                self.lower_weights[i, i] = 1
                self.upper_weights[i, i] = 1
               
            # some points are negative and some are positive
            else:
                if VERBOSE:
                    print("DeepPolyReluLayer forward: some points are negative and some are positive")
                
                # TODO: optimize alpha with gradient descent
                alpha = 1
                self.lower_weights[i, i] = alpha
                slope = u / (u - l)
                self.upper_weights[i, i] = slope
                
                # self.lower_bias = 0
                self.upper_bias[0, i] = slope * l
                

                lower_to_return[0, i] = alpha * l # self.lower[i, i] * lower[1, i]
                upper_to_return[0, i] = slope * u # self.upper[i, i] * upper[1, i]
            
        if VERBOSE:
            print("DeepPolyReluLayer forward:self.lower_weights shape %s, self.upper_weights shape %s" %
                  (str(self.lower_weights.shape), str(self.upper_weights.shape)))
            print("DeepPolyReluLayer: lower_bound shape %s, upper_bound shape %s, x shape %s" %
                  (str(lower_to_return.shape), str(upper_to_return.shape), str(x.shape)))
            
        assert lower_to_return.shape == upper_to_return.shape
        assert self.lower_bias.shape == self.upper_bias.shape
        assert self.lower_weights.shape == self.upper_weights.shape
        assert (lower_to_return <= upper_to_return).all()

        return x, lower_to_return, upper_to_return
