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
        self.weights = layer.weight
        self.bias = layer.bias

    # swap the bounds depending on the sign of the weight
    # return new lower and upper bounds
    @staticmethod
    def swap_and_forward(lower_bound, upper_bound, weights, bias):
        # mask = weights < 0
        # lower_bound = lower_bound.expand(mask.shape[0], -1)
        # upper_bound = upper_bound.expand(mask.shape[0], -1)
        # assert lower_bound.shape == upper_bound.shape == mask.shape
        # lower_bound[mask], upper_bound[mask] = - 1 * upper_bound[mask], -1 * lower_bound[mask]

        # new_lower_bound = torch.matmul(lower_bound, weights) + bias
        # new_upper_bound = torch.matmul(upper_bound, weights) + bias
        
        negative_mask = (weights < 0).int()
        positive_mask = (weights >= 0).int()

        negative_weights = torch.mul(negative_mask, weights)
        positive_weights = torch.mul(positive_mask, weights)

        new_lower_bound = torch.matmul(upper_bound, negative_weights.t(
        )) + torch.matmul(lower_bound, positive_weights.t()) + bias

        new_upper_bound = torch.matmul(lower_bound, negative_weights.t(
        )) + torch.matmul(upper_bound, positive_weights.t()) + bias


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
