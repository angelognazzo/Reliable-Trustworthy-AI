import torch
from settings import VERBOSE
import torch.nn.functional as F

class DeepPolyResnetBlock(torch.nn.Module):
    def __init__(self, l):
        super().__init__()
        
        self.block = l
        self.path_a = l.path_a
        self.path_b = l.path_b
        
        self.weights = None
        self.bias = None
        

    def forward(self, x):
       pass