import torch
import torch.nn.functional as F


def merge_conv_kernels(k1, k2):
    """
    :input k1: A tensor of shape ``(out1, in1, s1, s1)``
    :input k2: A tensor of shape ``(out2, in2, s2, s2)``
    :returns: A tensor of shape ``(out2, in1, s1+s2-1, s1+s2-1)``
      so that convolving with it equals convolving with k1 and
      then with k2.
    """
    padding = k2.shape[-1] - 1
    # Permute to adapt to BHCW
    k1 = k1.permute(1, 0, 2, 3)
    # Flip because this is actually correlation
    k2 = k2.flip(-1, -2)

    k3 = torch.conv2d(k1, k2, padding=padding)
    k3 = k3.permute(1, 0, 2, 3)
    return k3

################################################
# Test with no padding, no stride: IT WORKS
################################################
# create two random kernels
k1 = torch.randn([1, 1, 3, 3], dtype=torch.float64)
k2 = torch.rand([2, 1, 5, 5], dtype=torch.float64)
# merge the kernels
k3 = merge_conv_kernels(k1, k2)

# create a random image
img = torch.randn([1, 1, 20, 20], dtype=torch.float64)

# apply the two kernels subsequently
out1 = F.conv2d(F.conv2d(img, k1), k2)
# apply the merged kernel
out2 = F.conv2d(img, k3)

# make sure the two results are the same
print(out1.shape, out2.shape)
assert ((out1 - out2).abs().sum() < 1e-5), "Porco Dio"

# print(out1[0])
# print("@@@@@@@@@@@@@@\n")
# print(out2[0])
# print("@@@@@@@@@@@@@@\n")
# print(out1[0] == out2[0])

################################################
# Test with stride: IT DOESN'T WORK
################################################
# create two random kernels
k1 = torch.randn([1, 1, 3, 3], dtype=torch.float64)
k2 = torch.rand([2, 1, 5, 5], dtype=torch.float64)
# merge the kernels
k3 = merge_conv_kernels(k1, k2)

# create a random image
img = torch.randn([1, 1, 20, 20], dtype=torch.float64)

# apply the two kernels subsequently
out1 = F.conv2d(F.conv2d(img, k1, stride=(2, 2)), k2, stride=(2, 2))
# apply the merged kernel
out2 = F.conv2d(img, k3, stride=(5, 5))

# make sure the two results are the same
print(out1.shape, out2.shape)
assert ((out1 - out2).abs().sum() < 1e-5), "Porco Dio"
