import torch

VERBOSE = True

# return the lower and upper bounds of the infinity norm of the input image
class InfinityNormLayer(torch.nn.Module):

    def __init__(self, eps) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # lower = torch.clamp(inputs - pert, min=0.0).to(DEVICE)
        # upper = torch.clamp(inputs + pert, max=1.0).to(DEVICE)
        lower = torch.maximum(x - self.eps, torch.tensor(0))
        upper = torch.minimum(x + self.eps, torch.tensor(1))

        if VERBOSE:
            print("InfinityNormLayer: lower_bound shape %s, upper_bound shape %s", lower.shape, upper.shape)
        
        assert lower.shape == upper.shape
        assert lower.shape[0] == 1
        
        # return torch.flatten(lower), torch.flatten(upper)
        return lower, upper
    

class DeepPolyLinearLayer(torch.nn.Module):
    
    def __init__(self, net, layer) -> None:
        super().__init__()
        self.net = net
        self.layer = layer
        self.weights = layer.weight
        self.bias = layer.bias
        
        
    def forward(self, x, lower_bound, upper_bound):
        mask = self.weights < 0
        lower_bound[mask], upper_bound[mask] = - 1 * upper_bound[mask], -1 * lower_bound[mask]
        
        new_lower_bound = torch.matmul(lower_bound, self.weights) + self.bias
        new_upper_bound = torch.matmul(upper_bound, self.weights) + self.bias
        x = self.layer(x)
        
        if VERBOSE:
            print("DeepPolyLinearLayer: lower_bound shape %s, upper_bound shape %s, x shape %s", new_lower_bound.shape, new_upper_bound.shape, x.shape)

            
        assert new_lower_bound.shape == new_upper_bound.shape
        assert new_lower_bound.shape[0] == x.shape[0]
        
        return x, new_lower_bound, new_upper_bound



# ! TODO: implement
class DeepPolyReluLayer(torch.nn.Module):
    
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
    
    def forward(self, x, lower, upper):
        if VERBOSE:
            print("DeepPolyReluLayer: lower_bound shape %s, upper_bound shape %s, x shape %s", lower.shape, upper.shape, x.shape)
            
        assert lower.shape == upper.shape
        
        return x, lower, upper


class DeepPolyNetwork(torch.nn.Module):

    def __init__(self, net, eps) -> None:
        super().__init__()
        
        self.eps = eps
        self.net = net
        
        self.layers = [InfinityNormLayer(self.eps)]
        # create a custom layer for each layer in the original network skipping the flattening and normalization layers
        for i in range(2, len(self.net.layers)):
            l = self.net.layers[i]
            if type(l) == torch.nn.modules.linear.Linear:
                self.layers.append(DeepPolyLinearLayer(net, l))
            elif type(l) == torch.nn.modules.activation.ReLU:
                self.layers.append(DeepPolyReluLayer(...))
            else:
                print("ERROR: layer type not supported")
        
        if VERBOSE:
            print("DeepPolyNetwork: Created %s layers (Infinity norm layer included)", len(self.layers))
        
        assert len(self.layers) > 0
        assert len(self.layers) == len(self.net.layers) - 1
        
        self.lower_bounds_list = []
        self.upper_bounds_list = []
        self.activation_list = []
   
    # ! TODO: implement
    # perform backsubstitution to compute the lower and upper bounds of the input image
    def backsubstitution(self):
        lower_bound = None
        upper_bound = None
        
        return lower_bound, upper_bound
    
    def forward(self, x):
        # normalize and flatten the input image
        x = self.net.layers[0](x)
        x = self.net.layers[1](x)
        self.activation_list.append(x)
        
        assert x.shape[0] == 1
        
        # perturb the input image passing the input through the infinity norm layer
        lower_bound, upper_bound = self.layers[0](x)
        self.lower_bounds_list.append(lower_bound)
        self.upper_bounds_list.append(upper_bound)
        
        # perform the forward pass for each custom layer (skipping the infinity norm layer)
        for i, l in enumerate(self.layers(1, len(self.layers))):
            
            if VERBOSE:
                print("DeepPolyNetwork: Forward pass for layer %s", i)
                print("DeepPolyNetwork: layer of type %s", type(l))
            
            # perform the forward pass for the current layer
            x, lower_bound, upper_bound = l(x, lower_bound, upper_bound)
            
            # if l is a linear layer, perform backsubstitution
            if type(l) == DeepPolyLinearLayer:
                if VERBOSE:
                    print("DeepPolyNetwork: Performing backsubstitution")
                    
                lower_bound_tmp, upper_bound_tmp = self.backsubstitution()
                
                assert lower_bound_tmp.shape == lower_bound.shape
                assert upper_bound_tmp.shape == upper_bound.shape
                
                lower_bound = torch.maximum(lower_bound, lower_bound_tmp)
                upper_bound = torch.minimum(upper_bound, upper_bound_tmp)
            
            self.lower_bounds_list.append(lower_bound)
            self.upper_bounds_list.append(upper_bound)
            self.activation_list.append(x)
            
            
        if VERBOSE:
            print("DeepPolyNetwork: Forward pass completed")
            
        return x, self.lower_bounds_list, self.upper_bounds_list