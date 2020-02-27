import logging
import torch.nn as nn
import torchvision.models as models
from .resnet import DistillerBottleneck
import distiller
from torchvision.models.utils import load_state_dict_from_url


__all__ = ["resnet50_earlyexit", "resnet18_earlyexit"]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_exits_def(num_classes, arch):
    expansion = 1  # models.ResNet.BasicBlock.expansion
    early_exit_point_1 = "layer1.1.relu3" if arch == "resnet50" else "layer1.1.relu"
    early_exit_point_2 = "layer2.1.relu3" if arch == "resnet50" else "layer2.1.relu"
    if arch == "resnet50":
        exits_def = [("layer1.1.relu3", nn.Sequential(nn.Conv2d(256, 10, kernel_size=7, stride=2, padding=3, bias=True),
                                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                      nn.Flatten(),
                                                      nn.Linear(1960, num_classes))),
                                                      # distiller.modules.Print())),
                     ("layer2.1.relu3", nn.Sequential(nn.Conv2d(512, 12, kernel_size=7, stride=2, padding=3, bias=True),
                                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                      nn.Flatten(),
                                                      nn.Linear(588, num_classes)))]
    elif arch == "resnet18":
        exits_def = [("layer1.1.relu", nn.Sequential(nn.Conv2d(64, 10, kernel_size=7, stride=2, padding=3, bias=True),
                                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                     nn.Flatten(),
                                                     nn.Linear(1960, num_classes))),
                                                     # distiller.modules.Print())),
                     ("layer2.1.relu", nn.Sequential(nn.Conv2d(128, 12, kernel_size=7, stride=2, padding=3, bias=True),
                                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                     nn.Flatten(),
                                                     nn.Linear(588, num_classes)))]
    else:
        raise ValueError("Architecture must be either \"resnet18\" or \"resnet50\"")
    return exits_def


class ResNetEarlyExit(models.ResNet):
    def __init__(self, *args, **kwargs):
        self.frozen = kwargs["frozen"]
        del(kwargs["strict"])  # Quick-fix to avoid modifying out of project files
        del(kwargs["frozen"])  # Quick-fix to avoid modifying out of project files
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()

    def prepare_early_exists(self, arch):
        self.ee_mgr.attach_exits(self, get_exits_def(num_classes=1000, arch=arch))

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs

    def named_parameters(self, prefix='', recurse=True):
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if self.frozen and "branch_net" not in name and "fc" not in name:
                param.requires_grad = False
                continue
            yield param


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetEarlyExit(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            if kwargs.get("strict", True):
                raise e
            logging.warning("The checkpoint weight shapes do not exactly match the network definition ones.")
            logging.warning(str(e))
    model.prepare_early_exists(arch)
    # assert not pretrained
    model.test_parameter = True
    return model


def resnet50_earlyexit(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model, with early exit branches.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', DistillerBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet18_earlyexit(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    from torchvision.models.resnet import BasicBlock
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
