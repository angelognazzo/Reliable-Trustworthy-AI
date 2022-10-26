import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Sequence, Tuple, Type


class ResidualBlock(nn.Module):
    def __init__(
        self,
        path_a: nn.Sequential,
        path_b: nn.Sequential,
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.path_a = path_a
        self.path_b = path_b

    def forward(self, x: Tensor) -> Tensor:
        out = self.path_a(x) + self.path_b(x)
        return out


class BasicBlock(ResidualBlock, nn.Module):
    expansion = 1

    in_planes: int
    planes: int
    stride: int
    bn: bool
    kernel: int

    out_dim: int

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        bn: bool = True,
        kernel: int = 3,
        in_dim: int = -1,
    ) -> None:
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.bn = bn
        self.kernel = kernel

        kernel_size = kernel
        assert kernel_size in [1, 2, 3], "kernel not supported!"
        p_1 = 1 if kernel_size > 1 else 0
        p_2 = 1 if kernel_size > 2 else 0

        layers_b: List[nn.Module] = []
        layers_b.append(
            nn.Conv2d(
                in_planes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=p_1,
                bias=(not bn),
            )
        )
        _, _, in_dim = self._getShapeConv(
            (in_planes, in_dim, in_dim),
            (self.in_planes, kernel_size, kernel_size),
            stride=stride,
            padding=p_1,
        )

        if bn:
            layers_b.append(nn.BatchNorm2d(planes))
        layers_b.append(nn.ReLU())
        layers_b.append(
            nn.Conv2d(
                planes,
                self.expansion * planes,
                kernel_size=kernel_size,
                stride=1,
                padding=p_2,
                bias=(not bn),
            )
        )
        _, _, in_dim = self._getShapeConv(
            (planes, in_dim, in_dim),
            (self.in_planes, kernel_size, kernel_size),
            stride=1,
            padding=p_2,
        )
        if bn:
            layers_b.append(nn.BatchNorm2d(self.expansion * planes))
        path_b = nn.Sequential(*layers_b)

        layers_a: List[nn.Module] = [torch.nn.Identity()]
        if stride != 1 or in_planes != self.expansion * planes:
            layers_a.append(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=(not bn),
                )
            )
            if bn:
                layers_a.append(nn.BatchNorm2d(self.expansion * planes))
        path_a = nn.Sequential(*layers_a)
        self.out_dim = in_dim
        super(BasicBlock, self).__init__(path_a, path_b)

    def _getShapeConv(
        self,
        in_shape: Tuple[int, int, int],
        conv_shape: Tuple[int, ...],
        stride: int = 1,
        padding: int = 0,
    ) -> Tuple[int, int, int]:
        inChan, inH, inW = in_shape
        outChan, kH, kW = conv_shape[:3]

        outH = 1 + int((2 * padding + inH - kH) / stride)
        outW = 1 + int((2 * padding + inW - kW) / stride)
        return (outChan, outH, outW)

def getShapeConv(
    in_shape: Tuple[int, int, int],
    conv_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
) -> Tuple[int, int, int]:
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)


class ResNet(nn.Sequential):
    def __init__(
        self,
        block: Type[BasicBlock],
        in_ch: int = 3,
        num_stages: int = 1,
        num_blocks: int = 2,
        num_classes: int = 10,
        in_planes: int = 64,
        bn: bool = True,
        last_layer: str = "dense",
        in_dim: int = 32,
        stride: Optional[Sequence[int]] = None,
    ):
        layers: List[nn.Module] = []
        self.in_planes = in_planes
        if stride is None:
            stride = (num_stages + 1) * [2]

        layers.append(
            nn.Conv2d(
                in_ch,
                self.in_planes,
                kernel_size=3,
                stride=stride[0],
                padding=1,
                bias=not bn,
            )
        )

        _, _, in_dim = getShapeConv(
            (in_ch, in_dim, in_dim), (self.in_planes, 3, 3), stride=stride[0], padding=1
        )

        if bn:
            layers.append(nn.BatchNorm2d(self.in_planes))

        layers.append(nn.ReLU())

        for s in stride[1:]:
            block_layers, in_dim = self._make_layer(
                block,
                self.in_planes * 2,
                num_blocks,
                stride=s,
                bn=bn,
                kernel=3,
                in_dim=in_dim,
            )
            layers.append(block_layers)

        if last_layer == "dense":
            layers.append(nn.Flatten())
            layers.append(
                nn.Linear(self.in_planes * block.expansion * in_dim**2, 100)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Linear(100, num_classes))
        else:
            exit("last_layer type not supported!")

        super(ResNet, self).__init__(*layers)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_layers: int,
        stride: int,
        bn: bool,
        kernel: int,
        in_dim: int,
    ) -> Tuple[nn.Sequential, int]:
        strides = [stride] + [1] * (num_layers - 1)
        cur_dim: int = in_dim
        layers: List[nn.Module] = []
        for stride in strides:
            layer = block(self.in_planes, planes, stride, bn, kernel, in_dim=cur_dim)
            layers.append(layer)
            cur_dim = layer.out_dim
            layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers), cur_dim

