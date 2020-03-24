import torch
import logging
import torch.nn as nn
import torchvision.models as models
from distiller.models.imagenet.resnet import ResNet, ResNetChunkType
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


def get_exits_def(num_classes, arch, inference_type):
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
        layer1 = ("layer1.1.relu", nn.Sequential(nn.Conv2d(64, 10, kernel_size=7, stride=2, padding=3, bias=True),
                                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                     nn.Flatten(),
                                                     nn.Linear(1960, num_classes)))
        layer2 = ("layer2.1.relu", nn.Sequential(nn.Conv2d(128, 12, kernel_size=7, stride=2, padding=3, bias=True),
                                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                                     nn.Flatten(),
                                                     nn.Linear(588, num_classes)))
        if inference_type == ResNetChunkType.CHUNK_CLIENT:
            exits_def = [layer2]
        elif inference_type == ResNetChunkType.CHUNK_SERVER:
            exits_def = []
        elif inference_type == ResNetChunkType.FINAL_EXIT:
            exits_def = []
        elif inference_type == ResNetChunkType.EARLY_EXIT_1:
            exits_def = [layer1]
        elif inference_type == ResNetChunkType.EARLY_EXIT_2:
            exits_def = [layer2]
        else:
            exits_def = [layer1, layer2]
    else:
        raise ValueError("Architecture must be either \"resnet18\" or \"resnet50\"")
    return exits_def


class ResNetEarlyExit(ResNet):
    def __init__(self, *args, **kwargs):
        self.freezing_schedule = None if "freezing_schedule" not in kwargs or kwargs["freezing_schedule"] == "None" \
            else kwargs["freezing_schedule"]

        self.first_parameter_call = True

        self.inference_type = kwargs.get("inference_type", None)

        del(kwargs["strict"])  # Quick-fix to avoid modifying out of project files
        if "freezing_schedule" in kwargs:
            del(kwargs["freezing_schedule"])  # Quick-fix to avoid modifying out of project files
        super().__init__(*args, **kwargs)
        self.ee_mgr = distiller.EarlyExitMgr()

    def prepare_early_exists(self, arch, num_classes):
        """

        :param arch: Either resnet18 or resnet50
        :param num_classes: An int
        :param inference_type: If running inference, it specifies which early exits to attach, if any.
                               If None, it uses only the final exit
        :return:
        """
        self.ee_mgr.attach_exits(self, get_exits_def(num_classes=num_classes, arch=arch,
                                                     inference_type=self.inference_type))

    def forward(self, x):
        self.ee_mgr.delete_exits_outputs(self)
        # Run the input through the network (including exits)
        x = super().forward(x)
        outputs = self.ee_mgr.get_exits_outputs(self) + [x]
        return outputs

    def parameters(self, recurse=True):
        if self.first_parameter_call:
            print("PARAMETERS TO TRAIN")
        for name, param in self.named_parameters(recurse=recurse):
            if do_freeze_layer(name, self.freezing_schedule):
                param.requires_grad = False
                continue
            if self.first_parameter_call:
                print("\t- %s" % name)
            yield param
        self.first_parameter_call = False


def do_freeze_layer(name, freezing_schedule):
    if freezing_schedule is None:
        return False
    elif freezing_schedule == "stage_1":
        if name in {"conv1.weight", "bn1.weight", "bn1.bias"}:
            return False
        return "layer1" not in name
    elif freezing_schedule == "stage_2":
        return "layer2" not in name
    elif freezing_schedule == "stage_3":
        return "layer3" not in name and "layer4" not in name and name != "fc.weight"
    elif freezing_schedule == "only_branches":
        return "branch_net" not in name
    elif freezing_schedule == "single_training_finetunning":
        return "layer3" not in name and "layer4" not in name and name != "fc.weight"
    else:
        return False


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
    model.prepare_early_exists(arch, kwargs["num_classes"])
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
