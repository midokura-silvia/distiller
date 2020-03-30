"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

import torch.nn as nn
import math
import copy
import torch
import distiller
import numpy as np

from distiller.modules import BranchPoint


__all__ = ['mobilenetv3_duoli']

DEFAULT_CHECKPOINT_PATH = "/home/xavi/workspace/projects/distilled/models/mobilenet_v3/mobilenet_v3_finetuning___2020.03.24-091746/mobilenet_v3_finetuning_best.pth.tar"


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def get_early_exit_definition(early_exit_position, early_exit_version, num_classes):
    if early_exit_position == 1:
        if early_exit_version == "v1":
            # v1
            return [("features.4.conv.8", nn.Sequential(
                                            nn.Conv2d(40, 10, kernel_size=7, stride=2, padding=1, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            nn.Linear(360, num_classes)))]
        elif early_exit_version == "v2":
            # v2
            return [("features.4.conv.8", nn.Sequential(
                                            nn.Conv2d(40, 40, kernel_size=7, stride=1, padding=3, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Conv2d(40, 60, kernel_size=5, stride=1, padding=1, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            # nn.Linear(360, 720),
                                            h_swish(),
                                            nn.Linear(2160, num_classes)))]
        elif early_exit_version == "v3":
            # v3
            return [("features.4.conv.8", nn.Sequential(
                                            nn.Conv2d(40, 40, kernel_size=7, stride=2, padding=3, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            nn.Linear(1960, 1080),
                                            h_swish(),
                                            nn.Linear(1080, num_classes)))]
    elif early_exit_position == 2:
        if early_exit_version == "v1":
            # v1
            return [("features.7.conv.8", nn.Sequential(
                                            nn.Conv2d(80, 20, kernel_size=7, stride=2, padding=2, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            nn.Linear(180, num_classes)))]
        elif early_exit_version == "v2":
            # v2
            return [("features.7.conv.8", nn.Sequential(
                                            nn.Conv2d(80, 80, kernel_size=7, stride=2, padding=2, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            nn.Linear(720, 1080),
                                            nn.Linear(1080, num_classes)))]
        elif early_exit_version == "v3":
            # v3
            return [("features.7.conv.8", nn.Sequential(
                                            nn.Conv2d(80, 80, kernel_size=5, stride=1, padding=2, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Conv2d(112, 160, kernel_size=3, stride=1, padding=1, bias=True),
                                            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                                            h_swish(),
                                            nn.Flatten(),
                                            nn.Linear(540, num_classes)))]
    elif early_exit_position == 3:
        if early_exit_version == "v1":
            # v1
            return [("features.11.conv.8", nn.Sequential(nn.Conv2d(112, 60, kernel_size=7, stride=2, padding=2, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            nn.Linear(540, num_classes)))]
        elif early_exit_version == "v2":
            # v2
            return [("features.11.conv.8", nn.Sequential(nn.Conv2d(112, 60, kernel_size=7, stride=2, padding=1, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Flatten(),
                                            nn.Linear(540, 1080),
                                            h_swish(),
                                            nn.Linear(1080, num_classes)))]
        elif early_exit_version == "v3":
            # v3
            return [("features.11.conv.8", nn.Sequential(nn.Conv2d(112, 112, kernel_size=5, stride=1, padding=2, bias=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.Conv2d(112, 160, kernel_size=3, stride=1, padding=1, bias=True),
                                            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                                            h_swish(),
                                            nn.Flatten(),
                                            nn.Linear(160, num_classes)))]
    else:
        raise ValueError("The early_exit_position parameter must be either 1 or 2")


def _split_module_name(mod_name):
    name_parts = mod_name.split('.')
    parent = '.'.join(name_parts[:-1])
    node = name_parts[-1]
    return parent, node


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, mobilenet_early_exit_branch=None, num_classes=1000,
                 freezing_schedule=None, width_mult=1., **kwargs):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        self._is_early_exit = mobilenet_early_exit_branch is not None
        self.mobilenet_early_exit_branch = mobilenet_early_exit_branch
        self.num_classes = num_classes
        self.first_parameter_call = True
        self.freezing_schedule = freezing_schedule
        self.ee_mgr = distiller.EarlyExitMgr()

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        if self.mobilenet_early_exit_branch not in {"early_exit_1", "early_exit_2", "early_exit_3"}:
            print("Preparing a early exit network")
            # building last several layers
            self.conv = nn.Sequential(
                conv_1x1_bn(input_channel, _make_divisible(exp_size * width_mult, 8)),
                SELayer(_make_divisible(exp_size * width_mult, 8)) if mode == 'small' else nn.Sequential()
            )
            self.avgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                h_swish()
            )
            output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
            self.classifier = nn.Sequential(
                nn.Linear(_make_divisible(exp_size * width_mult, 8), output_channel),
                nn.BatchNorm1d(output_channel) if mode == 'small' else nn.Sequential(),
                h_swish(),
                nn.Linear(output_channel, num_classes),
                nn.BatchNorm1d(num_classes) if mode == 'small' else nn.Sequential(),
                h_swish() if mode == 'small' else nn.Sequential()
            )

        self._initialize_weights()

    def prepare_early_exit(self, early_exit_mode, early_exit_version):
        self.ee_mgr.attach_exits(self, get_early_exit_definition(early_exit_mode,
                                                                 early_exit_version,
                                                                 self.num_classes))

    def do_freeze_layer(self, name):
        if self.freezing_schedule is None:
            return False
        elif self.freezing_schedule == "until_branch_1":
            layer_num = int(name.split('.')[1])
            return layer_num > 4
        elif self.freezing_schedule in {"only_branch_1", "only_branch_2", "only_branch_3"}:
            return "branch_net" not in name
        elif self.freezing_schedule == "until_branch_2":
            layer_num = int(name.split('.')[1])
            return layer_num > 7
        elif self.freezing_schedule == "until_branch_3":
            layer_num = int(name.split('.')[1])
            return layer_num > 11
        else:
            return False

    def forward(self, x):
        x = self.features(x)
        if self.mobilenet_early_exit_branch not in {"early_exit_1", "early_exit_2", "early_exit_3"}:
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        if self.mobilenet_early_exit_branch in {"early_exit_1", "early_exit_2", "early_exit_3"}:
            return self.ee_mgr.get_exits_outputs(self) + self.ee_mgr.get_exits_outputs(self)
        if self._is_early_exit:
            return self.ee_mgr.get_exits_outputs(self) + [x]
        else:
            return x

    def parameters(self, recurse=True):
        if self.first_parameter_call:
            print("PARAMETERS TO TRAIN")
        for name, param in self.named_parameters(recurse=recurse):
            if self.do_freeze_layer(name):
                param.requires_grad = False
                continue
            if self.first_parameter_call:
                print("\t- %s" % name)
            yield param
        self.first_parameter_call = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_duoli(pretrained=False, mobilenet_early_exit_branch=None, **kwargs):
    mode = kwargs["mobilenet_mode"]
    if mode == "small":
        cfgs = [
            # k, t, c, SE, NL, s
            [3,  16,  16, 1, 0, 2],
            [3,  72,  24, 0, 0, 2],
            [3,  88,  24, 0, 0, 1],
            [5,  96,  40, 1, 1, 2],
            [5, 240,  40, 1, 1, 1],
            [5, 240,  40, 1, 1, 1],
            [5, 120,  48, 1, 1, 1],
            [5, 144,  48, 1, 1, 1],
            [5, 288,  96, 1, 1, 2],
            [5, 576,  96, 1, 1, 1],
            [5, 576,  96, 1, 1, 1],
        ]
    else:

        if mobilenet_early_exit_branch in {None, "trunk_early_exit_1", "trunk_early_exit_2", "trunk_early_exit_3"}:
            cfgs = [
                # k, t, c, SE, NL, s
                [3,  16,  16, 0, 0, 1],
                [3,  64,  24, 0, 0, 2],
                [3,  72,  24, 0, 0, 1],
                [5,  72,  40, 1, 0, 2],
                [5, 120,  40, 1, 0, 1],
                [5, 120,  40, 1, 0, 1],
                [3, 240,  80, 0, 1, 2],
                [3, 200,  80, 0, 1, 1],
                [3, 184,  80, 0, 1, 1],
                [3, 184,  80, 0, 1, 1],
                [3, 480, 112, 1, 1, 1],
                [3, 672, 112, 1, 1, 1],
                [5, 672, 160, 1, 1, 1],
                [5, 672, 160, 1, 1, 2],
                [5, 960, 160, 1, 1, 1]
            ]
        elif mobilenet_early_exit_branch == "early_exit_1":
            cfgs = [
                # k, t, c, SE, NL, s
                [3,  16,  16, 0, 0, 1],
                [3,  64,  24, 0, 0, 2],
                [3,  72,  24, 0, 0, 1],
                [5,  72,  40, 1, 0, 2],
            ]
        elif mobilenet_early_exit_branch == "early_exit_2":
            cfgs = [
                # k, t, c, SE, NL, s
                [3,  16,  16, 0, 0, 1],
                [3,  64,  24, 0, 0, 2],
                [3,  72,  24, 0, 0, 1],
                [5,  72,  40, 1, 0, 2],
                [5, 120,  40, 1, 0, 1],
                [5, 120,  40, 1, 0, 1],
                [3, 240,  80, 0, 1, 2],
            ]
        elif mobilenet_early_exit_branch == "early_exit_3":
            cfgs = [
                # k, t, c, SE, NL, s
                [3,  16,  16, 0, 0, 1],
                [3,  64,  24, 0, 0, 2],
                [3,  72,  24, 0, 0, 1],
                [5,  72,  40, 1, 0, 2],
                [5, 120,  40, 1, 0, 1],
                [5, 120,  40, 1, 0, 1],
                [3, 240,  80, 0, 1, 2],
                [3, 200,  80, 0, 1, 1],
                [3, 184,  80, 0, 1, 1],
                [3, 184,  80, 0, 1, 1],
                [3, 480, 112, 1, 1, 1]
            ]

        model = MobileNetV3(cfgs, mode=mode, mobilenet_early_exit_branch=mobilenet_early_exit_branch, **kwargs)
        get_early_exit_definition = None

        if mobilenet_early_exit_branch in {"early_exit_1", "trunk_early_exit_1"}:
            model.prepare_early_exit(1, kwargs["mobilenet_early_exit_branch_version"])
        elif mobilenet_early_exit_branch in {"early_exit_2", "trunk_early_exit_2"}:
            model.prepare_early_exit(2, kwargs["mobilenet_early_exit_branch_version"])
        elif mobilenet_early_exit_branch in {"early_exit_3", "trunk_early_exit_3"}:
            model.prepare_early_exit(3, kwargs["mobilenet_early_exit_branch_version"])

        for name, module in model.named_modules():
            print(name)

        if pretrained:
            checkpoint = torch.load(DEFAULT_CHECKPOINT_PATH, map_location=lambda storage, loc: storage)
            checkpoint["state_dict"] = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
            list_new_pairs = []
            for exit_point in model.ee_mgr.exit_points:
                for key, value in checkpoint["state_dict"].items():
                    if exit_point in key:
                        list_new_pairs.append((exit_point + ".branched_module." + key.split('.')[-1], value))
            for key, value in list_new_pairs:
                checkpoint["state_dict"][key] = value

            anomalous_keys = model.load_state_dict(checkpoint['state_dict'], False)

            import logging
            msglogger = logging.getLogger()
            if anomalous_keys:
                # This is pytorch 1.1+
                missing_keys, unexpected_keys = anomalous_keys
                if unexpected_keys:
                    msglogger.warning("Warning: the loaded checkpoint contains the following unexpected state keys %s" %
                                      str(unexpected_keys))
                if missing_keys:
                    for missing_key in missing_keys:
                        is_exit_point = False
                        for exit_point in model.ee_mgr.exit_points:
                            if exit_point in missing_key:
                                is_exit_point = True
                        if not is_exit_point:
                            raise ValueError("The loaded checkpoint is missing %d state keys" %
                                             len(missing_keys))
                    msglogger.warning("Warning: The loaded checkpoint is missing the following keys: %s" %
                                      str(missing_keys))

    return model

