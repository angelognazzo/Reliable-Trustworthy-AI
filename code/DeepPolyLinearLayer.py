import torch
from settings import VERBOSE


class DeepPolyLinearLayer(torch.nn.Module):
    """
    Class implementing the LinearLayer of the DeepPoly algorithm
    """

    def __init__(self, net, layer) -> None:
        super().__init__()
        self.net = net
        self.layer = layer
        # we want 784x50 and not the opposite
        self.weights = layer.weight.t()
        # we want 1x50 and not the opposite
        self.bias = layer.bias.reshape(1, -1)

    # swap the bounds depending on the sign of the weights
    # return new lower and upper bounds
    @staticmethod
    def swap_and_forward(lower_bound, upper_bound, weights, bias):
        
        negative_mask = (weights < 0).int()
        positive_mask = (weights >= 0).int()

        negative_weights = torch.mul(negative_mask, weights)
        positive_weights = torch.mul(positive_mask, weights)

        new_lower_bound = torch.matmul(upper_bound, negative_weights) + torch.matmul(lower_bound, positive_weights) + bias

        new_upper_bound = torch.matmul(lower_bound, negative_weights) + torch.matmul(upper_bound, positive_weights) + bias


        if VERBOSE:
            print("DeepPolyLinearLayer swap_and_forward: lower_bound shape %s, upper_bound shape %s" % (new_lower_bound.shape, new_upper_bound.shape))
        
        assert new_lower_bound.shape == new_upper_bound.shape, "swap_and_forward: lower and upper bounds have different shapes"
        assert (new_lower_bound <= new_upper_bound).all(), "swap_and_forward: error with the box bounds: lower > upper"

        return new_lower_bound, new_upper_bound

    # forward pass through the network
    def forward(self, x, lower_bound, upper_bound):
        new_lower_bound, new_upper_bound = self.swap_and_forward(
            lower_bound, upper_bound, self.weights, self.bias)
        x = self.layer(x)
        
        if VERBOSE:
            print("DeepPolyLinearLayer: x shape %s" % (str(x.shape)))

        assert new_lower_bound.shape[0] == x.shape[0]

        return x, new_lower_bound, new_upper_bound
