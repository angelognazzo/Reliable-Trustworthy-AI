import torch
import torch.nn.functional as F

def convmatrix2d(kernel, image_shape, padding: int=0):
    """
    kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
    image: (in_channels, image_height, image_width, ...)
    padding: assumes the image is padded with ZEROS of the given amount
    in every 2D dimension equally. The actual image is given with unpadded dimension.
    """
    stride = 1
    # If we want to pad, request a bigger matrix as the kernel will convolve
    # over a bigger image.
    if padding:
        old_shape = image_shape
        pads = (padding, padding, padding, padding)
        image_shape = (image_shape[0], image_shape[1] + padding*2, image_shape[2]
                       + padding*2)
    else:
        image_shape = tuple(image_shape)
    assert image_shape[0] == kernel.shape[1]
    assert len(image_shape[1:]) == len(kernel.shape[2:])
    assert stride == 1

    kernel = kernel.to('cpu') # always perform the below work on cpu

    result_dims = (torch.tensor(image_shape[1:]) -
                   torch.tensor(kernel.shape[2:]))//stride + 1
    mat = torch.zeros((kernel.shape[0], *result_dims, *image_shape))
    for i in range(mat.shape[1]):
        for j in range(mat.shape[2]):
            mat[:,i,j,:,i:i+kernel.shape[2],j:j+kernel.shape[3]] = kernel
    mat = mat.flatten(0, len(kernel.shape[2:])).flatten(1)

    # Handle zero padding. Effectively, the zeros from padding do not
    # contribute to convolution output as the product at those elements is zero.
    # Hence the columns of the conv mat that are at the indices where the
    # padded flattened image would have zeros can be ignored. The number of
    # rows on the other hand must not be altered (with padding the output must
    # be larger than without). So..

    # We'll handle this the easy way and create a mask that accomplishes the
    # indexing
    if padding:
        mask = torch.nn.functional.pad(torch.ones(old_shape), pads).flatten()
        mask = mask.bool()
        mat = mat[:, mask]

    return mat

#####################################################
# Use toepiz matrix TOO SLOW (there is padding, but no stride)
###################################################
# img = torch.randn([1, 1, 32, 32], dtype=torch.float64)
# k1 = torch.randn([50, 1, 3, 3], dtype=torch.float64)
# k2 = torch.rand([250, 50, 5, 5], dtype=torch.float64)
# w1 = convmatrix2d(k1, (k1.shape[1], img.shape[2], img.shape[3]))
# w2 = convmatrix2d(k2, (k1.shape[0], img.shape[2] - k1.shape[2] + 1, img.shape[3] - k1.shape[3] + 1))
# print(w1.shape)
# print(w2.shape)
# print(F.conv2d(img, k1).shape)
# # apply the merged kernel
# img_flatten = img.flatten().reshape(-1, 1).float()
# # print(img_flatten.shape)
# # wx = torch.matmul(w2, w1).float()
# # print(wx.shape)
# # out2 = torch.matmul(wx, img_flatten).t()
# out2 = torch.matmul(w2, torch.matmul(w1, img_flatten))
# print(out2.shape)
# output_shape = (1, k2.shape[0], img.shape[2] - k1.shape[2] + 1 - k2.shape[2] + 1, img.shape[3] - k1.shape[3] + 1 - k2.shape[3] + 1)
# out2 = out2.reshape(output_shape)
#
# # apply the two kernels subsequently
# out1 = F.conv2d(F.conv2d(img, k1), k2)
#
# print((out1 - out2).abs().sum())
# assert ((out1 - out2).abs().sum() < 1), "Porco Dio"
# exit()



# ##############################################################################
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

    k3 = torch.conv2d(k1, k2, padding=padding + 1)
    k3 = k3.permute(1, 0, 2, 3)
    return k3



def add_padding_to_kernel(k, padding):
    return k

def add_stride_to_kernel(k, stride):
    return k

def adjust_kernel(k,padding,stride):
    return add_padding_to_kernel(add_stride_to_kernel(k))

# padding = (2,2)
# stride = 1
#
# img = torch.randn([1, 1, 20, 20], dtype=torch.float64)
# k1 = torch.randn([1, 1, 3, 3], dtype=torch.float64)
# k2 = torch.rand([1, 1, 5, 5], dtype=torch.float64)
# # merge the kernels
# k1_with_padding = adjust_kernel(k1, padding, stride)
# k2_with_padding = add_padding_to_kernel(k2,  padding, stride)
# print(k3.shape)
#
# out1 = F.conv2d(F.conv2d(img, k1, padding=padding), k2)
# print(out1.shape)
#
# out2 = F.conv2d(F.conv2d(img, k1_with_padding), k2_with_padding)
# # make sure the two results are the same
# print(out1.shape, out2.shape)
# print((out1 - out2).abs().sum())
# assert ((out1 - out2).abs().sum() < 1), "Porco Dio"


#
# ################################################
# # Test with padding: IT DOESN'T WORK
# ################################################
# create two random kernels
k1 = torch.randn([1, 1, 3, 3], dtype=torch.float64)
k2 = torch.rand([1, 1, 5, 5], dtype=torch.float64)
# merge the kernels
k3 = merge_conv_kernels(k1, k2)
print(k3.shape)

# create a random image
img = torch.randn([1, 1, 20, 20], dtype=torch.float64)

# apply the two kernels subsequently
# il primo padding non può mai essere 1.
# il secondo non può essere 1 se il primo padding è 0 o 1
out1 = F.conv2d(F.conv2d(img, k1, padding=(2,2)), k2, padding=(1,1))
out2 = F.conv2d(F.conv2d(img, k1, padding=(3,3)), k2)
print(out1.shape)
print(out2.shape)

# make sure the two results are the same
# print(out1.shape, out2.shape)
print((out1 - out2).abs().sum())
assert ((out1 - out2).abs().sum() < 1), "Porco Dio"

# # apply the merged kernel
# out2 = F.conv2d(img, k3, padding=(3, 3))
# print(out2.shape)

# # make sure the two results are the same
# # print(out1.shape, out2.shape)
# print((out1 - out2).abs().sum())
# assert ((out1 - out2).abs().sum() < 1), "Porco Dio"





























################################################
# Test with no padding, no stride: IT WORKS
################################################
# create two random kernels
# k1 = torch.randn([1, 1, 3, 3], dtype=torch.float64)
# k2 = torch.rand([2, 1, 5, 5], dtype=torch.float64)
# # merge the kernels
# k3 = merge_conv_kernels(k1, k2)
#
# # create a random image
# img = torch.randn([1, 1, 20, 20], dtype=torch.float64)
#
# # apply the two kernels subsequently
# out1 = F.conv2d(F.conv2d(img, k1), k2)
# # apply the merged kernel
# out2 = F.conv2d(img, k3)
#
# # make sure the two results are the same
# print(out1.shape, out2.shape)
# assert ((out1 - out2).abs().sum() < 1e-5), "Porco Dio"
#
# # print(out1[0])
# # print("@@@@@@@@@@@@@@\n")
# # print(out2[0])
# # print("@@@@@@@@@@@@@@\n")
# # print(out1[0] == out2[0])
