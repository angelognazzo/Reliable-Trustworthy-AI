import torch

W_a = torch.rand(10, 10)
W_b = torch.rand(10, 10)

W = W_a + W_b

lower = torch.rand(10, 1)

tmp1 = torch.matmul(W_a, lower) + torch.matmul(W_b, lower)
tmp2 = torch.matmul(W, lower)


