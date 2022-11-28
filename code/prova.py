import torch

input_shape = [1, 1, 4, 4]
a = torch.eye(16)
b = a.view(input_shape + [16])
b = b.permute(0, 1, 4, 2, 3)
print(b)
kernel = torch.randn([1, 1, 3,3])
# print(kernel)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Input shape: " + str(input_shape))
print("Output shape: ")
print("kernel shape: " + str(kernel.shape))
print("W shape: " + str(b.shape))

w = torch.conv3d(b, kernel.unsqueeze(
    2), stride=(1, 1, 1)).permute(0, 1, 3, 4, 2)
# print(w)
print("After Convolutional: " + str(w.shape))
# print(w)
w = w[0]
weights = w.reshape(16, 4).t()
# print(weights)
# print(weights.shape)