import torch
class DeepPolyAlphaLoss(torch.nn.Module):
    def __init__(self):
        super(DeepPolyAlphaLoss, self).__init__()
        
    def forward(self, lower_bound):
        
        lower_bound = lower_bound.squeeze()
        
        return torch.nn.functional.relu(-lower_bound).sum().log()