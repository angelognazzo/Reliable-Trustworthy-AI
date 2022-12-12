import torch
from settings import VERBOSE


class DeepPolyReluLayer(torch.nn.Module):
    """
    Class implementing the ReluLayer of the DeepPoly algorithm
    """

    def __init__(self, layer, in_features) -> None:
        super().__init__()
        self.layer = layer
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None

        in_features = in_features[0] * in_features[1] * in_features[2]
        self.alpha = torch.nn.Parameter(data=torch.ones(in_features), requires_grad=True)
        # self.alpha = None
        
    def forward(self, x, lower, upper, input_shape):
        
        lower = torch.clone(lower)
        upper = torch.clone(upper)
        
        lower_to_return = torch.clone(lower)
        upper_to_return = torch.clone(upper)
        if VERBOSE:
            print("DeepPolyReluLayer forward: lower shape %s, upper shape %s" % (str(lower_to_return.shape), str(upper_to_return.shape)))
        
        assert lower_to_return.shape == upper_to_return.shape, "DeepPolyReluLayer forward: lower_to_return and upper bounds have different shapes"
        
        # compute the relu on the input
        x = self.layer(x)

        mask_points_negative = upper_to_return <= 0
        mask_points_positive = lower_to_return >= 0
        mask_points_negative_and_positive = torch.logical_and((lower_to_return < 0), (upper_to_return > 0))
        
        # all the points are negative
        lower_to_return = torch.where(mask_points_negative, torch.tensor(0.), lower_to_return)
        upper_to_return = torch.where(mask_points_negative, torch.tensor(0.), upper_to_return)
        # lower_w = torch.where(mask_points_negative, 0., 0.)
        # upper_w = torch.where(mask_points_negative, 0., 0.)
        # self.lower_bias = torch.where(mask_points_negative, 0., 0.)
        # self.upper_bias = torch.where(mask_points_negative, 0., 0.)
        lower_w = torch.zeros_like(lower_to_return)
        upper_w = torch.zeros_like(upper_to_return)
        self.lower_bias = torch.zeros_like(lower_to_return)
        self.upper_bias = torch.zeros_like(upper_to_return)
        
        # all the points are positive
        # lower_to_return = torch.where(mask_points_positive, lower_to_return, lower_to_return)
        # upper_to_return = torch.where(mask_points_positive, upper_to_return, upper_to_return)
        lower_w = torch.where(mask_points_positive, torch.tensor(1.), lower_w)
        upper_w = torch.where(mask_points_positive, torch.tensor(1.), upper_w)
        # self.lower_bias = torch.where(mask_points_positive, 0., 0.)
        # self.upper_bias = torch.where(mask_points_positive, 0., 0.)
        
        # in between
        slope = torch.where(mask_points_negative_and_positive, torch.div(upper_to_return, upper_to_return-lower_to_return), torch.tensor(0.))
        lower_to_return = torch.where(mask_points_negative_and_positive, self.alpha*lower_to_return, lower_to_return)
        upper_to_return = torch.where(mask_points_negative_and_positive, slope*upper_to_return, upper_to_return)
        lower_w = torch.where(mask_points_negative_and_positive, self.alpha, lower_w)
        upper_w = torch.where(mask_points_negative_and_positive, slope, upper_w)
        # self.lower_bias = torch.where(mask_points_negative_and_positive, 0., 0.)
        self.upper_bias = torch.where(mask_points_negative_and_positive, slope*lower_to_return, torch.zeros_like(lower_to_return))
        
        
        self.lower_weights=torch.diag(lower_w.squeeze())
        self.upper_weights = torch.diag(upper_w.squeeze())
        
        assert lower_to_return.shape == upper_to_return.shape
        assert self.lower_bias.shape == self.upper_bias.shape
        assert self.lower_weights.shape == self.upper_weights.shape
        assert (lower_to_return <= upper_to_return).all()
        
        return x, lower_to_return, upper_to_return, input_shape
