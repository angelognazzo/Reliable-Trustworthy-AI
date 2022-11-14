import torch
from settings import VERBOSE


class DeepPolyLinearLayer(torch.nn.Module):

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
        mask = weights < 0
        lower_bound[mask], upper_bound[mask] = - 1 * \
            upper_bound[mask], -1 * lower_bound[mask]

        new_lower_bound = torch.matmul(lower_bound, weights) + bias
        new_upper_bound = torch.matmul(upper_bound, weights) + bias

        if VERBOSE:
            print("DeepPolyLinearLayer swap_and_forward: lower_bound shape %s, upper_bound shape %s", new_lower_bound.shape, new_upper_bound.shape)

        assert new_lower_bound.shape == new_upper_bound.shape, "swap_and_forward: lower and upper bounds have different shapes"
        assert (new_lower_bound <= new_upper_bound).all(), "swap_and_forward: error with the box bounds: lower > upper"

        return new_lower_bound, new_upper_bound

    # forward pass through the network
    def forward(self, x, lower_bound, upper_bound):
        new_lower_bound, new_upper_bound = self.swap_and_forward(
            lower_bound, upper_bound, self.weights, self.bias)
        x = self.layer(x)
        
        if VERBOSE:
            print("DeepPolyLinearLayer: x shape %s", x.shape)

        assert new_lower_bound.shape[0] == x.shape[0]

        return x, new_lower_bound, new_upper_bound
