import torch
from settings import VERBOSE
from backsubstitution import backsubstitution


class DeepPolyFinalVerificationLayer(torch.nn.Module):
    """
    Class implementing the final verifaction layer of the DeepPoly algorithm
    """

    def __init__(self, previous_layers, true_label) -> None:
        super().__init__()
        self.previous_layers = previous_layers
        self.in_features = 10
        self.out_features = 9
        self.true_label = true_label
        
        self.isRes=False
        
        A = torch.zeros((self.in_features, self.in_features))
        A[:, self.true_label] = 1
        A = -1 * torch.eye(self.in_features) + A

        A = torch.cat((A[:self.true_label], A[self.true_label+1:]))
        b = torch.zeros(self.in_features - 1)
        
        self.lower_weights = A.t()
        self.upper_weights = A.t()

        self.lower_bias = b
        self.upper_bias = b
        
    # forward pass through the network
    def forward(self, lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag):
        lower_bound, upper_bound, _, _, _, _ = backsubstitution(self.previous_layers + [self], first_lower_bound, first_upper_bound)

        return lower_bound, upper_bound
