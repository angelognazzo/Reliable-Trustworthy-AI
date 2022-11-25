import torch

# in_shape = [1, 3, 3]

# w = torch.eye(9).view(in_shape + [9])
# # print(w.shape)
# # print(w)
# w = w.unsqueeze(0).permute(0,1,4,2,3)
# print(w.shape)
# print(w)


# a = torch.rand([3, 3])
# a = a.reshape(1, 3, 3)
# print(a[0].shape)

# a = torch.eye(3)
# a = a.reshape(1, 3, 3)
# print(a[0].shape)


a = torch.rand([1, 10, 32, 32])
print(a.shape[:-1])
b = a.flatten(start_dim=1, end_dim=-1)
b = b.reshape(1, 10, 32, 32)

assert (a == b).all()
