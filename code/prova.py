import torch

torch.manual_seed(0)
shape = (1, 3, 3, 3)
m = torch.nn.BatchNorm2d(shape[1])
x = torch.randn(shape)
output = m(x)

m.eval()
print("gamma: ", m.weight, m.weight.shape)
print("beta: ", m.bias, m.bias.shape)
print("mean: ", m.running_mean, m.running_mean.shape)
print("var: ", m.running_var, m.running_var.shape)
print("eps: ", m.eps)
output = m(x)
print(x)
print(output)


var = m.running_var.reshape(-1, 1)  # (C, 1)
var = var.repeat(1, shape[2] * shape[3])  # (C, H*W)
var = var.flatten()  # (C*H*W, )
var = torch.sqrt(var + m.eps)  # sqrt(var_i + eps)
var = 1 / var
gamma = m.weight.reshape(-1, 1)
gamma = gamma.repeat(1, shape[2] * shape[3])
gamma = gamma.flatten()

w = torch.diag(gamma * var)

mean = m.running_mean.reshape(-1, 1)  # (C, 1)
mean = mean.repeat(1, shape[2] * shape[3])  # (C, H*W)
mean = mean.flatten()  # (C*H*W, )
bias = mean * var * gamma
bias = bias + m.bias.reshape(-1, 1).repeat(1, shape[2] * shape[3]).flatten()

bound = x.flatten()
output2 = torch.matmul(w, bound) - bias

print(output.flatten())
print(output2)
exit()
mean = m.running_mean.reshape(-1, 1) # (C, 1)
mean = mean.repeat(1, shape[2] * shape[3]) # (C, H*W)
mean = mean.flatten() # (C*H*W, )
bound = x.flatten()
mean = 1 - mean / bound # 1 - mean_i / x_i

var = m.running_var.reshape(-1, 1)  # (C, 1)
var = var.repeat(1, shape[2] * shape[3])  # (C, H*W)
var = var.flatten()  # (C*H*W, )
var = torch.sqrt(var + m.eps)  # sqrt(var_i + eps)
# print(var)

w = torch.diag(mean / var)
bound = bound.reshape(-1, 1)
output2 = torch.matmul(w, bound)


print(output.flatten())
print(output2)
assert torch.allclose(output.flatten(), output2)




exit()
# With Learnable Parameters
m = torch.nn.BatchNorm2d(2)
# Without Learnable Parameters
#m = torch.nn.BatchNorm2d(100, affine=False)
input = torch.randn(1, 2, 3, 4)
output = m(input)
#print(output.shape)
#print(output)
"""print("gamma: ", m.weight, m.weight.shape)
print("beta: ", m.bias, m.bias.shape)
print("mean: ", m.running_mean, m.running_mean.shape)
print("var: ", m.running_var, m.running_var.shape)
print("eps: ", m.eps)"""

#print("mean: ", m.running_mean, m.running_mean.shape)
print("gamma: ", m.weight, m.weight.shape)
# alternative

sample_mean = input.mean(axis=0)
sample_var = input.var(axis=0)

print("sample_mean", sample_mean.shape)
print("sample_var", sample_var.shape)



var_sqrt = torch.sqrt(sample_var + m.eps)#m.running_var 

w_prime= m.weight / var_sqrt
w_prime2= torch.div(m.weight, var_sqrt)
assert torch.allclose(w_prime, w_prime2)
print(w_prime.shape)
b = (- sample_mean * m.weight) / var_sqrt + m.bias #m.running_mean
prob=torch.ones(1, input.shape[0]*input.shape[2]*input.shape[3])
w= torch.matmul(w_prime.reshape(-1, 1), prob).flatten()
print(w.shape)
weights = (torch.diag(w))
print("w", weights.shape)
bias = b.repeat(input.shape[0]*input.shape[2]*input.shape[3])
print("input", input.flatten().shape)
output2= torch.matmul(weights, input.flatten()) + bias


print(output2)
print(output.flatten())
assert torch.allclose(output.flatten(), output2)

