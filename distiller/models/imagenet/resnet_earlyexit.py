import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import BasicBlock


__all__ = ['resnet50_earlyexit', "resnet18_earlyexit"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNetEarlyExit(models.ResNet):

    def __init__(self, block, layers, num_classes=1000, pretrained=False):
        super(ResNetEarlyExit, self).__init__(block, layers, num_classes)

        multiplier = 4 if block.expansion == 1 else 1
        multiplier2 = 4

        if pretrained:
            self.load_state_dict(model_zoo.load_url(models.resnet.model_urls['resnet18']))

        # Define early exit layers
        self.conv1_exit0 = nn.Conv2d(int(256/multiplier), 50, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2_exit0 = nn.Conv2d(50, 12, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv1_exit1 = nn.Conv2d(int(512/multiplier), 12, kernel_size=5, stride=2, padding=3, bias=True)
        self.fc_exit0 = nn.Linear(147 * multiplier2, num_classes)
        self.fc_exit1 = nn.Linear(192 * multiplier2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # Add early exit layers
        exit0 = nn.functional.avg_pool2d(x, 2)
        exit0 = self.conv1_exit0(exit0)
        exit0 = self.conv2_exit0(exit0)

        exit0 = exit0.view(exit0.size(0), -1)
        exit0 = self.fc_exit0(exit0)

        x = self.layer2(x)

        # Add early exit layers
        exit1 = nn.functional.avg_pool2d(x, 2)
        exit1 = self.conv1_exit1(exit1)
        exit1 = exit1.view(exit1.size(0), -1)
        exit1 = self.fc_exit1(exit1)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return a list of probabilities
        output = list()
        output.append(exit0)
        output.append(exit1)
        output.append(x)
        return output


def resnet50_earlyexit(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetEarlyExit(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(models.resnet.model_urls['resnet50']))
    return model


#def resnet26_earlyexit(pretrained=False, **kwargs):
#    """Constructs a ResNet-26 model.
#    """
#    model = ResNetEarlyExit(Bottleneck, [2, 2, 2, 2], **kwargs)
#    return model


def resnet18_earlyexit(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEarlyExit(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
