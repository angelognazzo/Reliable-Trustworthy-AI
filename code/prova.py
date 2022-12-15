import torch


i = 5
for i in range(i, -1, -1):
    print(i)


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

