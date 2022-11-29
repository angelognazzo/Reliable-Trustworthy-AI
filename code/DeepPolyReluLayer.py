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
        
    def forward(self, x, lower, upper, input_shape):

        if VERBOSE:
            print("DeepPolyReluLayer forward: lower shape %s, upper shape %s" %
                  (str(lower.shape), str(upper.shape)))

        assert lower.shape == upper.shape, "DeepPolyReluLayer forward: lower and upper bounds have different shapes"

        # compute the relu on the actual input
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
            
            # all the points are negative, everything is clipped to zero
            # since weights are already inzitialized at 0 (see above) we don't have to do anything
            if u <= 0:
                if VERBOSE:
                    pass
                    # print("DeepPolyReluLayer forward: all the points are negative")
                pass

            # all the points are positive
            elif l >= 0:
                if VERBOSE:
                    pass
                    # print("DeepPolyReluLayer forward: all the points are positive")
                lower_to_return[0, i] = l
                upper_to_return[0, i] = u
                self.lower_weights[i, i] = 1
                self.upper_weights[i, i] = 1

            # some points are negative and some are positive
            else:
                if VERBOSE:
                    pass
                    # print("DeepPolyReluLayer forward: some points are negative and some are positive")

                # TODO: optimize alpha with gradient descent
                alpha = 1
                self.lower_weights[i, i] = alpha
                slope = u / (u - l)
                self.upper_weights[i, i] = slope

                # self.lower_bias = 0
                self.upper_bias[0, i] = slope * l

                # self.lower[i, i] * lower[1, i]
                lower_to_return[0, i] = alpha * l
                # self.upper[i, i] * upper[1, i]
                upper_to_return[0, i] = slope * u

        if VERBOSE:
            print("DeepPolyReluLayer forward:self.lower_weights shape %s, self.upper_weights shape %s" %
                  (str(self.lower_weights.shape), str(self.upper_weights.shape)))
            print("DeepPolyReluLayer: lower_bound shape %s, upper_bound shape %s, x shape %s" %
                  (str(lower_to_return.shape), str(upper_to_return.shape), str(x.shape)))

        assert lower_to_return.shape == upper_to_return.shape
        assert self.lower_bias.shape == self.upper_bias.shape
        assert self.lower_weights.shape == self.upper_weights.shape
        assert (lower_to_return <= upper_to_return).all()

        return x, lower_to_return, upper_to_return, input_shape



    def forward_optimized(self, x, lower, upper, input_shape):
        
        if VERBOSE:
            print("DeepPolyReluLayer forward: lower shape %s, upper shape %s" % (str(lower.shape), str(upper.shape)))
        
        assert lower.shape == upper.shape, "DeepPolyReluLayer forward: lower and upper bounds have different shapes"
        
        x_tmp, lower_tmp, upper_tmp = self.old_forward(x.clone(), lower.clone(), upper.clone())
        
        # compute the relu on the input
        x = self.layer(x)
        input_shape = x.shape

        # TODO: use torch.where instead of for loops
        # all the points are negative
        lower = torch.where(upper <= 0, torch.zeros_like(lower), lower)
        upper = torch.where(upper <= 0,  torch.zeros_like(upper), upper)
        lower_w = torch.where(upper <= 0, 0, 0)
        upper_w = torch.where(upper <= 0, 0, 0)
        self.lower_bias = torch.where(upper <= 0, 0, 0)
        self.upper_bias = torch.where(upper <= 0, 0, 0)
        
        # all the points are positive
        lower = torch.where(lower >= 0, upper, lower)
        upper = torch.where(lower >= 0, upper, upper)
        lower_w = torch.where(lower >= 0, torch.zeros_like(lower_w), lower_w)
        upper_w = torch.where(lower >= 0, torch.zeros_like(upper_w), upper_w)
        self.lower_bias = torch.where(lower >= 0, 0, 0)
        self.upper_bias = torch.where(lower >= 0, 0, 0)
        
        # in between
        alpha = 2
        tmp = alpha * torch.ones_like(lower_w)
        tmp = tmp.type(torch.float)
        print(tmp)
        slope = torch.where((upper - lower) != 0, torch.div(upper, upper-lower), torch.zeros_like(upper))
        lower = torch.where((lower < 0) & (upper > 0), alpha*lower, lower)
        upper = torch.where((lower < 0) & (upper > 0), torch.mul(slope, upper), upper)
        lower_w = torch.where((lower < 0) & (upper > 0), tmp, lower_w)
        upper_w = torch.where((lower < 0) & (upper > 0), slope * torch.ones_like(upper_w), upper_w)
        self.lower_bias = torch.where((lower < 0) & (upper > 0), 0, 0)
        self.upper_bias = torch.where((lower < 0) & (upper > 0), torch.mul(slope, lower), torch.zeros_like(lower))
        
        
        # TODO: this diag should not always be here. FIX IT
        self.lower_weights=torch.diag(lower_w)
        self.upper_weights=torch.diag(upper_w)

        assert lower.shape == upper.shape
        assert self.lower_bias.shape == self.upper_bias.shape
        assert self.lower_weights.shape == self.upper_weights.shape
        assert (lower <= upper).all()
        
        assert (lower_tmp == lower).all()
        assert (upper_tmp == upper).all()
        assert (x_tmp == x).all()
        
        return x, lower, upper, input_shape
