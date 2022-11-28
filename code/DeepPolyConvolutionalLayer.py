
import torch
from settings import VERBOSE
import torch.nn.functional as F

class DeepPolyConvolutionalLayer(torch.nn.Module):
    """
    Class implementing the ConvolutionalLayer of the DeepPoly algorithm
    """
    
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer
        self.kernel = layer.weight
        self.weights = None 
        
        # The bias is a 1D tensor, we want to reshape it to a 2D tensor??? TODO: check this
        self.bias_kernel = layer.bias
        self.bias = None
        
        self.stride = layer.stride
        self.padding = layer.padding
        
        self.output_shape = None
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer: weights shape: %s, bias shape %s, stride %s, padding %s" % (
                str(self.kernel.shape), str(self.bias_kernel.shape), str(self.stride), str(self.padding)))

    # swap the bounds depending on the sign of the weights
    # return new lower and upper bounds
    def swap_and_forward(self, lower_bound, upper_bound, weights, bias, stride, padding):
        negative_mask = (weights < 0).int()
        positive_mask = (weights >= 0).int()

        negative_weights = torch.mul(negative_mask, weights)
        positive_weights = torch.mul(positive_mask, weights)
 
        lower_bound_new = F.conv2d(upper_bound, negative_weights, bias, stride, padding) + \
            F.conv2d(lower_bound, positive_weights, bias, stride, padding) + bias.reshape(1, -1, 1, 1)

        upper_bound_new = F.conv2d(lower_bound, negative_weights, bias, stride, padding) + \
            F.conv2d(upper_bound, positive_weights, bias, stride, padding) + bias.reshape(1, -1, 1, 1)
            
        assert lower_bound_new.shape == upper_bound_new.shape, "swap_and_forward: lower and upper bounds have different shapes"
        assert (lower_bound_new <= upper_bound_new).all(), "swap_and_forward: error with the box bounds: lower > upper"

        return lower_bound_new.flatten(start_dim=1, end_dim=-1), upper_bound_new.flatten(start_dim=1, end_dim=-1)
    
    
    def forward(self, x, lower_bound, upper_bound, input_shape):
        # out_height = (input_shape[2] + 2 * self.padding[1] - self.kernel.shape[2]) // self.stride[1] + 1
        # out_width = (input_shape[1] + 2 * self.padding[0] - self.kernel.shape[1]) // self.stride[0] + 1
        # self.output_shape = (1, self.kernel_shape[0], out_height, out_width)
        
        # x, lower_bound and upper_bound are flattened (i.e. [1, 3072]), we want to reshape them to being a tensor so that we can perfrom the convolutions(i.e [1, 3, 32, 32])
        x = x.reshape(input_shape)
        lower_bound = lower_bound.reshape(input_shape)
        upper_bound = upper_bound.reshape(input_shape)
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer RESHAPE: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))
        
        lower_bound, upper_bound = self.swap_and_forward(
            lower_bound, upper_bound, self.kernel, self.bias_kernel, self.stride, self.padding)
        
        if VERBOSE:
            print("DeepPolyConvolutionalLayer: x shape %s, lower_bound shape %s, upper_bound shape %s" % (
                str(x.shape), str(lower_bound.shape), str(upper_bound.shape)))
        
        x = self.layer(x)

        # weight matrix is of shape [n_elemnts_input, n_elements_output]
        # self.weights = torch.zeros([input_shape.numel(), x.shape.numel()])
        # print(self.weights.shape)
        # for out_z in range(x.shape[1]):
        #     for out_x in range(x.shape[3]):
        #         for out_y in range(x.shape[2]):
        #             considered_pixel = x[:, out_z, out_y, out_x]
        #             # for element [out_z, out_y, out_x] of the output, we want to know which column of the weight matrix we are going to update
        #             index_weights_column = (((out_y * x.shape[2] * x.shape[3] + out_x * x.shape[3]) / x.shape[2] + 1) + (out_z * x.shape[3] * x.shape[2])) - 1
        #             index_weights_column = int(index_weights_column)
        #             for channel_z_input in range(input_shape[1]):
        #                 for x_shift in range(self.kernel.shape[3]):
        #                     for y_shift in range(self.kernel.shape[2]):
        #                         # fill the column "index_weights_column" of the weight matrix
        #                         # what is a? la prima parentesi restituisce un numero  tra  1 e 4 (eg matrice 2*2 con elementi
        #                         # chiamati 1 2 3 4 in senso orario
        #                         # prendi elemento 1,0 della matrice (ovvero il 3): avremo che  (1*2*2 + 0*2)/2+1=3)
        #                         # poi ci sommiamo il canale in cui ci troviamo dell'output: la nostra matrice weights ha tante colonne
        #                         # quanti numero pixel output di un channel * numero channels: quindi se abbiamo matrice  2*2 con 2 canali
        #                         # avremo 4*2=8 colonne e le prime 4 sono corrispondenti ai 4 pixel del primo canale
                                
                                
        #                         # una volta nella colonna, cerchiamo la posizione giusta da riempire.
        #                         # se abbiamo kernel 3*3=[A,B,C,D,E,F,G,H,I] allora la colonna sara qualcosa tipo [ABC 0 DEF 0 GHI 0 0 0 0]
        #                         #prima di tutto mettiamo zeri tra ABC DEF etc, sono tanti quanti i pixel non presi dal kernel in direzione delle
        #                         # colonne. poi la posizione dipende da quale elemento della tripletta stiamo considerando (0,1,2) e da quale delle
        #                         # tre triplette abbiamo scelto. poi sommiamo anche l'effetto della traslazione dovuto ad essere in un channel
        #                         # dell'input. Infine la stride: quando stiamo considerando (scorriamo sulle colonne dell'output) la prima colonna 
        #                         # dell-output facciamo salti in vertical col kernel (stride verticale), altrimenti consideriamo solo quelli 
        #                         # accumulati finora, per gli altri pixel della colonna output facciamo salti in orizzontale (piu quelli 
        #                         # verticali fatti finora)
                                
                                
        #                         how_many_zeros_between = y_shift * (input_shape[3] - self.kernel.shape[3])
        #                         # take care of horizzontal padding
        #                         if self.padding[1] - out_x > 0 or self.padding[1] - (x.shape[3] - out_x - 1) > 0:
        #                             how_many_zeros_between += max(0, self.padding[1] - out_x)
        #                         # take care of vertical padding
        #                         if self.padding[2] - out_y > 0 or self.padding[2] - (x.shape[2] - out_y - 1) > 0:
        #                             position_in_column = x_shift + (max(y_shift, y_shift + self.padding[2] - out_y)  * self.kernel.shape[2]) + how_many_zeros_between
        #                         else:
        #                             position_in_column = x_shift + (y_shift * self.kernel.shape[2]) + how_many_zeros_between
                               
        #                         # add channel information
        #                         position_in_column += channel_z_input * input_shape[2] * input_shape[3]
        #                         # take care of vertical stride
        #                         position_in_column += self.stride[0] * out_y if (out_x == 0) else  self.stride[0] * (max(0, out_y - 1))
        #                         # take care of horizzontal stride
        #                         position_in_column += self.stride[1] * out_x if (out_x != 0) else 0

        #                         x_shit_prime = x_shift
        #                         if self.padding[1] - out_x > 0:
        #                             x_shit_prime += self.padding[1]
        #                         elif self.padding[1] - (x.shape[3] - out_x - 1) > 0:
        #                             x_shit_prime -= self.padding[1]
                                
        #                         y_shift_prime = y_shift
        #                         if self.padding[2] - out_y > 0:
        #                             y_shift_prime += self.padding[2]
        #                         elif self.padding[2] - (x.shape[2] - out_y - 1) > 0:
        #                             y_shift_prime -= self.padding[2]
                                    

        #                         self.weights[int(position_in_column), index_weights_column] = self.kernel[channel_z_input, out_z, y_shift, x_shift]
        
        #################################################################
        #################################################################
        #################################################################
        w = torch.eye(input_shape.numel()).view(list(input_shape) + [input_shape.numel()])
        w = w.permute(0, 1, 4, 2, 3)

        w = F.conv3d(w, self.kernel.unsqueeze(2), stride=tuple(
            [1] + list(self.stride)), padding=tuple([0] + list(self.padding))).permute(0, 1, 3, 4, 2)
        
        # remove the first empty dimension
        w = w[0]
        self.weights = torch.flatten(w, start_dim=0, end_dim=2).t()

        b = torch.ones(w.shape[:-1]) * self.bias_kernel[:, None, None]
        self.bias = torch.flatten(b)

        #################################################################
        #################################################################
        #################################################################
        
        input_shape = x.shape
        x = x.flatten(start_dim=1, end_dim=-1)

        assert lower_bound.shape == x.shape == upper_bound.shape, "swap_and_forward CNN: lower and upper bounds have different shapes"
        assert (lower_bound <= upper_bound).all(), "swap_and_forward CNN: error with the box bounds: lower > upper"
          
        return x, lower_bound, upper_bound, input_shape