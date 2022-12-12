import torch
class DeepPolyAlphaLoss(torch.nn.Module):
    def __init__(self):
        super(DeepPolyAlphaLoss, self).__init__()
        
    def forward(self, lower_bounds, upper_bounds, target):
        
        lower_bounds = lower_bounds.squeeze()
        upper_bounds = upper_bounds.squeeze()
        
        target_lower_bound = lower_bounds[target]

        g = target_lower_bound.repeat(10) - upper_bounds
        g[target] = 0
        
        return -torch.sum(g)