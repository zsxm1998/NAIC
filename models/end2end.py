import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Callable, Union, Optional, List, Any


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        n_channels: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetEncoder, self).__init__()
        self.n_channels = n_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ) if stride == 1 else nn.Sequential(nn.AvgPool2d(stride, stride, ceil_mode=True),
                conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self._forward_impl(x), dim=1)


def resnet_encoder18(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet_encoder34(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet_encoder50(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet_encoder101(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet_encoder152(**kwargs: Any) -> ResNetEncoder:
    return ResNetEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet_encoder(model_depth: int, **kwargs: Any) -> ResNetEncoder:
    assert model_depth in [18, 34, 50, 101, 152], f'model_depth={model_depth}'
    if model_depth == 18:
        model = resnet_encoder18(**kwargs)
    elif model_depth == 34:
        model = resnet_encoder34(**kwargs)
    elif model_depth == 50:
        model = resnet_encoder50(**kwargs)
    elif model_depth == 101:
        model = resnet_encoder101(**kwargs)
    elif model_depth == 152:
        model = resnet_encoder152(**kwargs)
    return model


resneet_encoder_out_dim_dir = {18:512, 34:512, 50:2048, 101:2048, 152:2048}


class Encoder(nn.Module):
    def __init__(self, intermediate_dim, input_dim):
        super(Encoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.input_dim = input_dim

        self.relu = nn.ReLU(inplace=True)
        self.el1 = nn.Linear(input_dim, input_dim)
        self.el2 = nn.Linear(input_dim, intermediate_dim)
        self.en2 = nn.BatchNorm1d(intermediate_dim, affine=False)

    def forward(self, x):
        return self.en2(self.el2(self.relu(self.el1(x))))


class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl1 = nn.Linear(intermediate_dim, 512)
        self.dn1 = nn.BatchNorm1d(512)
        self.dl2 = nn.Linear(512, 1024)
        self.dn2 = nn.BatchNorm1d(1024)
        self.dl3 = nn.Linear(1024, 2048)
        self.dn3 = nn.BatchNorm1d(2048)
        self.dl4 = nn.Linear(2048, 4096)
        self.dn4 = nn.BatchNorm1d(4096)
        self.dl5 = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.relu(self.dn3(self.dl3(x)))
        x = self.relu(self.dn4(self.dl4(x)))
        out = self.dl5(x)
        return out


class MoCo(nn.Module):
    def __init__(self, model_depth, dim=128, K=65536, m=0.999):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m

        self.extractor_q = resnet_encoder(model_depth)
        self.extractor_k = resnet_encoder(model_depth)
        self.extractor_out_dim = resneet_encoder_out_dim_dir[model_depth]
        self.compress_dim = dim
        for param_q, param_k in zip(self.extractor_q.parameters(), self.extractor_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.encoder_q = Encoder(dim, self.extractor_out_dim)
        self.encoder_k = Encoder(dim, self.extractor_out_dim)
        self.decoder = Decoder(dim, self.extractor_out_dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_label", torch.ones(K)*-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.extractor_q.parameters(), self.extractor_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_label):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr+batch_size <= self.K:
            self.queue[ptr:ptr+batch_size] = keys
            self.queue_label[ptr:ptr+batch_size] = keys_label
        else:
            diff = self.K - ptr
            self.queue[ptr:] = keys[:diff]
            self.queue_label[ptr:] = keys_label[:diff]
            self.queue[:batch_size-diff] = keys[diff:]
            self.queue_label[:batch_size-diff] = keys_label[diff:]
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, input_q, input_k, k_label):
        q = self.extractor_q(input_q)
        q_comp = self.encoder_q(q)
        q_reco = self.decoder(q_comp.half().float())
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.extractor_k(input_k)
            k_comp = self.encoder_k(k)
        k_reco = self.decoder(k_comp.half().float())
        q_comp = F.normalize(q_comp, dim=1)
        k_comp = F.normalize(k_comp, dim=1)
        queue = self.queue.clone().detach()
        queue_label = self.queue_label.clone().detach()
        self._dequeue_and_enqueue(k_comp, k_label)
        return q, k, q_comp, k_comp, queue, queue_label, q_reco, k_reco