import torch


i = 5
for i in range(i, -1, -1):
    print(i)


# With Learnable Parameters
m = torch.nn.BatchNorm2d(100)
# Without Learnable Parameters
#m = torch.nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)
print(output.shape)
print(output)
print("gamma: ", m.weight, m.weight.shape)
print("beta: ", m.bias, m.bias.shape)
print("mean: ", m.running_mean, m.running_mean.shape)
print("var: ", m.running_var, m.running_var.shape)
print("eps: ", m.eps)
