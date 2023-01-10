import torch
from InfinityNormLayer import InfinityNormLayer
from DeepPolyLinearLayer import DeepPolyLinearLayer
from DeepPolyReluLayer import DeepPolyReluLayer
from DeepPolyConvolutionalLayer import DeepPolyConvolutionalLayer
from DeepPolyResnetBlock import DeepPolyResnetBlock
from DeepPolyBatchNormLayer import DeepPolyBatchNormLayer
from DeepPolyFinalVerificationLayer import DeepPolyFinalVerificationLayer
from settings import VERBOSE
from utils import compute_out_dimension
import networks
import resnet


class DeepPolyNetwork(torch.nn.Module):
    """
    Perform a DeepPoly analysis on the given network
    """
    def __init__(self, net, eps, true_label) -> None:
        super().__init__()

        self.eps = eps
        
        self.infinity_norm_layer = InfinityNormLayer(self.eps)
        # contains the custom layers of the network
        layers = []
        
        # parse the network to understand which layers to create
        # in case of a resnet, parsing is more complicated because layers are nested in blocks
        layers_to_create = []
        if type(net) == networks.NormalizedResnet:
            
            # get the correct normalization layer
            self.normalization_layer = net.normalization
            #  get the actual resnet from the 'normalized resnet' object
            dataset = net.dataset
            net = net.resnet
            # get all the layers of the resnet
            l = list(net.modules())
            i = 1
            while i < len(l):
                
                module = l[i]
                # we don't care about sequentials blocks. go to the next layer
                if type(module) == torch.nn.modules.Sequential:
                    i += 1
                    continue
                # append the current layer
                layers_to_create.append(module)
                
                # if the current layer is of type 'BasicBlock' I want to get the layer as a whole and skip
                # all the layers inside the block
                if type(module) == resnet.BasicBlock:
                    i += len(module.path_a) + len(module.path_b) + 2 + 1
                else:
                    i += 1
        # in case of a normal network, parsing is easier, just get the correct field
        else:
            dataset = net.dataset
            layers_to_create = net.layers
            self.normalization_layer = layers_to_create[0]

        if dataset == 'mnist':
            out_dimension = (1, 28, 28)
        elif dataset == 'cifar10':
            out_dimension = (3, 32, 32)
        else:
            raise("DeepPolyNetwork constructor ERROR: dataset not supported")
        
        # create a custom layer for each layer in the original network
        for i in range(0, len(layers_to_create)):
            l = layers_to_create[i]
            
            # skip the flattening layer and the normalization.
            # These layers are not present in every network, For example in CNNs and ResNets. That's why this check is necessary
            # we dont want to create a custom layer for these layers, we just resuse the torch implementation
            if type(l) == torch.nn.modules.flatten.Flatten or type(l) == networks.Normalization:
                continue
            
            out_dimension_tmp = compute_out_dimension(out_dimension, l)
            # create a custom layer based on the type of the original layer
            if type(l) == torch.nn.modules.linear.Linear:
                layers.append(DeepPolyLinearLayer(l, layers.copy()))
            elif type(l) == torch.nn.modules.activation.ReLU:
                layers.append(DeepPolyReluLayer(l, layers.copy(), out_dimension_tmp))
            elif type(l) == torch.nn.modules.Conv2d:
                layers.append(DeepPolyConvolutionalLayer(l, layers.copy(), out_dimension))
            elif type(l) == resnet.BasicBlock:
                layers.append(DeepPolyResnetBlock(l, layers.copy(), out_dimension))
            elif type(l) == torch.nn.modules.BatchNorm2d:
                layers.append(DeepPolyBatchNormLayer(l, layers.copy(), out_dimension))
            else:
                raise Exception("DeepPolyNetwork constructor ERROR: layer type not supported")
            
            out_dimension = out_dimension_tmp
            
        layers.append(DeepPolyFinalVerificationLayer(layers.copy(), true_label))
        if VERBOSE:
            print("DeepPolyNetwork: Created %s layers (Infinity norm layer included)" % (len(layers)))

        #assert len(layers) > 0, "DeepPolyNetwork constructor: no layers created"

        # this field is used to let torch see the paramters to optimize of the custom layers (alpha of RELU)
        self.sequential_layers = torch.nn.Sequential(*layers)
   
    def forward(self, x):
        # perturb the input image passing the input through the infinity norm custom layer
        lower_bound, upper_bound = self.infinity_norm_layer(x)
        
        # normalize the input image and flatten
        lower_bound = self.normalization_layer(lower_bound).flatten().reshape(1, -1)
        upper_bound = self.normalization_layer(upper_bound).flatten().reshape(1, -1)

        # save the initial lower and upper bounds
        first_lower_bound = torch.clone(lower_bound)
        first_upper_bound = torch.clone(upper_bound)

        # input dimensions should be the same even after our transformations
        #assert lower_bound.shape == upper_bound.shape, "DeepPolyNetwork forward: input shape mismatch after normalization and flattening"
        if VERBOSE:
            print("DeepPolyNetwork forward: shape after normalization and flattening: lower bound %s, upper bound %s" % (lower_bound.shape, upper_bound.shape))

        # perform the forward pass for each custom layer
        for i in range(len(self.sequential_layers)):
            l=self.sequential_layers[i]
            # ! perform the FORWARD pass for the current layer
            #print(i)
            if i+1<=len(self.sequential_layers)-1:
                if type(self.sequential_layers[i+1])==DeepPolyReluLayer or type(self.sequential_layers[i+1])==DeepPolyFinalVerificationLayer:
                    flag=True
                else: flag=False
            elif type(l)==DeepPolyFinalVerificationLayer:
                flag=True
            else:
                flag=False
            lower_bound, upper_bound = l(lower_bound, upper_bound, first_lower_bound, first_upper_bound, flag)

        return lower_bound, upper_bound
