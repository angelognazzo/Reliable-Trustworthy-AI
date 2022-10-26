import torch
import torch.nn as nn
from resnet import ResNet, BasicBlock

class Normalization(nn.Module):

    def __init__(self, device, dataset):
        super(Normalization, self).__init__()
        if dataset == 'mnist':
            self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
            self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)
        elif dataset == 'cifar10':
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1)).to(device)
            self.sigma = torch.FloatTensor([0.2023, 0.1994, 0.201]).view((1, 3, 1, 1)).to(device)
        else:
            assert False

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnected(nn.Module):

    def __init__(self, device, dataset, input_size, input_channels, fc_layers, act='relu'):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device, dataset), nn.Flatten()]
        prev_fc_size = input_size * input_size * input_channels
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                if act == 'relu':
                    layers += [nn.ReLU()]
                else:
                    assert False
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv(nn.Module):

    def __init__(self, device, dataset, input_size, input_channels, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device, dataset)]
        prev_channels = input_channels
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NormalizedResnet(nn.Module):

    def __init__(self, device, resnet):
        super(NormalizedResnet, self).__init__()
        
        self.normalization = Normalization(device, 'cifar10')
        self.resnet = resnet

    def forward(self, x):
        x = self.normalization(x)
        x = self.resnet(x)
        return x


def get_net_name(net):
    net_names = {
        'net1': 'net1_mnist_fc1.pt',
        'net2': 'net2_mnist_fc2.pt',
        'net3': 'net3_cifar10_fc3.pt',
        'net4': 'net4_mnist_conv1.pt',
        'net5': 'net5_mnist_conv2.pt',
        'net6': 'net6_cifar10_conv2.pt',
        'net7': 'net7_mnist_conv3.pt',
        'net8': 'net8_cifar10_resnet_2b.pt',
        'net9': 'net9_cifar10_resnet_2b2_bn.pt',
        'net10': 'net10_cifar10_resnet_4b.pt',
    }
    return net_names[net]
    
def get_network(device, net):
    if net == 'net1':
        return FullyConnected(device, 'mnist', 28, 1, [50, 10])
    elif net == 'net2':
        return FullyConnected(device, 'mnist', 28, 1, [100, 50, 10])
    elif net == 'net3':
        return FullyConnected(device, 'cifar10', 32, 3, [100, 100, 10])
    elif net == 'net4':
        return Conv(device, 'mnist', 28, 1, [(16, 3, 2, 1)], [100, 10], 10)
    elif net == 'net5':
        return Conv(device, 'mnist', 28, 1, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10)
    elif net == 'net6':
        return Conv(device, 'cifar10', 32, 3, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10)
    elif net == 'net7':
        return Conv(device, 'mnist', 28, 1, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10)
    elif net == 'net8':
        return ResNet(BasicBlock, num_stages=1, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
    elif net == 'net9':
        return ResNet(
            BasicBlock, num_stages=2, num_blocks=1, in_planes=16,
            bn=True, last_layer="dense", stride=[2, 2, 2])
    elif net == 'net10':
        return ResNet(BasicBlock, num_stages=2, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
    assert False

