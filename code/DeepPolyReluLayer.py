import torch
from settings import VERBOSE
#from backsubstitution import backsubstitution


class DeepPolyReluLayer(torch.nn.Module):
    """
    Class implementing the ReluLayer of the DeepPoly algorithm
    """

    def __init__(self, layer, previous_layers, in_features) -> None:
        super().__init__()
        # self.layer = layer
        self.previous_layers = previous_layers
        
        self.lower_weights = None
        self.upper_weights = None
        self.lower_bias = None
        self.upper_bias = None
        
        self.isRes=False

        in_features = in_features[0] * in_features[1] * in_features[2]  
        self.alpha = torch.nn.Parameter(data=0.2*torch.randn(in_features), requires_grad=True)


    def forward(self, current_lower, current_upper, first_lower_bound, first_upper_bound, flag):
       
        if VERBOSE:
            print("DeepPolyReluLayer forward: lower shape %s, upper shape %s" % (str(current_lower.shape), str(current_upper.shape)))
    
        # mask_points_negative = current_upper <= 0
        mask_points_positive = current_lower >= 0
        mask_points_negative_and_positive = torch.logical_and((current_lower < 0), (current_upper > 0))
        
        # all the points are negative
        lower_w = torch.zeros_like(current_lower)
        upper_w = torch.zeros_like(current_upper)
        self.lower_bias = torch.zeros_like(current_lower)
        self.upper_bias = torch.zeros_like(current_upper)
        lower_bound=torch.zeros_like(current_lower)
        upper_bound=torch.zeros_like(current_upper)
        
        # all the points are positive
        lower_w = torch.where(mask_points_positive, torch.tensor(1.), lower_w)
        upper_w = torch.where(mask_points_positive, torch.tensor(1.), upper_w)
        lower_bound = torch.where(mask_points_positive, current_lower, torch.clone(lower_bound))
        upper_bound = torch.where(mask_points_positive, current_upper, torch.clone(upper_bound))
        
        # in between
        slope = torch.where(mask_points_negative_and_positive, torch.div(current_upper, current_upper-current_lower), torch.tensor(0.))
        lower_w = torch.where(mask_points_negative_and_positive, torch.sigmoid(self.alpha), lower_w)
        upper_w = torch.where(mask_points_negative_and_positive, slope, upper_w)
        self.upper_bias = torch.where(mask_points_negative_and_positive, -slope*current_lower, torch.zeros_like(current_lower))
        self.lower_weights = torch.diag(lower_w.squeeze())
        self.upper_weights = torch.diag(upper_w.squeeze())
        lower_bound = torch.where(mask_points_negative_and_positive, torch.mul(torch.sigmoid(self.alpha), current_lower), torch.clone(lower_bound))
        upper_bound= torch.where(mask_points_negative_and_positive, torch.mul(slope, current_upper), torch.clone(upper_bound))

        
        #lower_bound, upper_bound, _, _, _, _ = backsubstitution(self.previous_layers + [self], first_lower_bound, first_upper_bound)


        #assert lower_bound.shape == upper_bound.shape
        #assert self.lower_bias.shape == self.upper_bias.shape
        #assert self.lower_weights.shape == self.upper_weights.shape
        #assert (lower_bound <= upper_bound).all()
        
        return lower_bound, upper_bound
