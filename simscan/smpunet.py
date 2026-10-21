"""U-Net из segmentation_models_pytorch (Unet с кодировщиком ResNet) на чистом torch: чекпойнты,
обученные с smp.Unet(...), грузятся без smp и timm.

Архитектура восстанавливается по формам весов: блоки ResNet (BasicBlock - resnet18/34,
Bottleneck - resnet50/101/152) и их число, каналы декодера, ядро головы. Прямой проход - как в
smp 0.5: признаки [x, conv1, layer1, ..., layer4], декодер от layer4 вверх, на каждом шаге -
повышение «nearest» до размера следующего признака, склейка с ним, две свёртки 3x3 + BN + ReLU;
голова - одна свёртка. Сверено с smp поэлементно (tests/test_smpunet.py).

Регрессионных голов у такой сети нет: если классов 8 - выход только sem; если 13 - первые 8
каналов sem, остальные 5 - как у PlanNet (dist, orient x2, heights x2).
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

N_CLASSES = 8


def is_smp_unet(sd: dict) -> bool:
    return "encoder.conv1.weight" in sd and "decoder.blocks.0.conv1.0.weight" in sd and "segmentation_head.0.weight" in sd


def _bn(sd: dict, key: str) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(sd[f"{key}.weight"].shape[0])


def _conv(sd: dict, key: str, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    w = sd[f"{key}.weight"]
    k = w.shape[-1]
    return nn.Conv2d(w.shape[1], w.shape[0], k, stride, k // 2, bias=bias)


class _Block(nn.Module):
    """BasicBlock / Bottleneck torchvision (ключи conv1, bn1, ..., downsample.0/1)."""

    def __init__(self, sd: dict, key: str, stride: int):
        super().__init__()
        self.bottleneck = f"{key}.conv3.weight" in sd
        self.conv1 = _conv(sd, f"{key}.conv1", 1 if self.bottleneck else stride)
        self.bn1 = _bn(sd, f"{key}.bn1")
        self.conv2 = _conv(sd, f"{key}.conv2", stride if self.bottleneck else 1)
        self.bn2 = _bn(sd, f"{key}.bn2")
        if self.bottleneck:
            self.conv3 = _conv(sd, f"{key}.conv3")
            self.bn3 = _bn(sd, f"{key}.bn3")
        self.downsample = None
        if f"{key}.downsample.0.weight" in sd:
            self.downsample = nn.Sequential(_conv(sd, f"{key}.downsample.0", stride), _bn(sd, f"{key}.downsample.1"))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.bottleneck:
            y = self.bn3(self.conv3(F.relu(y)))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class _Encoder(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        self.conv1 = _conv(sd, "encoder.conv1", 2)
        self.bn1 = _bn(sd, "encoder.bn1")
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        for li in range(1, 5):
            n = 1 + max(int(m.group(1)) for k in sd if (m := re.match(rf"encoder\.layer{li}\.(\d+)\.", k)))
            setattr(self, f"layer{li}", nn.Sequential(*[_Block(sd, f"encoder.layer{li}.{b}", 2 if li > 1 and b == 0 else 1)
                                                         for b in range(n)]))

    def forward(self, x):
        f = [x]
        x = F.relu(self.bn1(self.conv1(x)))
        f.append(x)
        x = self.layer1(self.maxpool(x))
        f.append(x)
        for layer in (self.layer2, self.layer3, self.layer4):
            x = layer(x)
            f.append(x)
        return f


def _conv_relu(sd: dict, key: str) -> nn.Sequential:
    return nn.Sequential(_conv(sd, f"{key}.0"), _bn(sd, f"{key}.1"), nn.ReLU(inplace=True))


class _DecoderBlock(nn.Module):
    def __init__(self, sd: dict, key: str):
        super().__init__()
        self.conv1 = _conv_relu(sd, f"{key}.conv1")
        self.conv2 = _conv_relu(sd, f"{key}.conv2")

    def forward(self, x, size, skip=None):
        x = F.interpolate(x, size=size, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv2(self.conv1(x))


class _Decoder(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        n = 1 + max(int(m.group(1)) for k in sd if (m := re.match(r"decoder\.blocks\.(\d+)\.", k)))
        if n != 5:
            raise ValueError(f"smp Unet: блоков декодера {n}, поддержана глубина 5")
        if any(k.startswith("decoder.center.") for k in sd):
            raise ValueError("smp Unet с center-блоком (add_center_block) не поддержан")
        self.blocks = nn.ModuleList(_DecoderBlock(sd, f"decoder.blocks.{i}") for i in range(n))

    def forward(self, feats):
        shapes = [f.shape[2:] for f in feats][::-1]
        f = feats[1:][::-1]
        x, skips = f[0], f[1:]
        for i, blk in enumerate(self.blocks):
            x = blk(x, shapes[i + 1], skips[i] if i < len(skips) else None)
        return x


class SmpUnet(nn.Module):
    """smp.Unet(ResNet) по state_dict; выход - словарь голов, как у PlanNet."""

    def __init__(self, sd: dict):
        super().__init__()
        self.encoder = _Encoder(sd)
        self.decoder = _Decoder(sd)
        w = sd["segmentation_head.0.weight"]
        self.segmentation_head = nn.Sequential(nn.Conv2d(w.shape[1], w.shape[0], w.shape[-1], padding=w.shape[-1] // 2))
        self.in_channels, self.classes = int(sd["encoder.conv1.weight"].shape[1]), int(w.shape[0])

    def describe(self) -> str:
        blocks = [len(getattr(self.encoder, f"layer{i}")) for i in range(1, 5)]
        kind = "Bottleneck" if self.encoder.layer1[0].bottleneck else "Basic"
        name = {((2, 2, 2, 2), "Basic"): "resnet18", ((3, 4, 6, 3), "Basic"): "resnet34",
                ((3, 4, 6, 3), "Bottleneck"): "resnet50", ((3, 4, 23, 3), "Bottleneck"): "resnet101",
                ((3, 8, 36, 3), "Bottleneck"): "resnet152"}.get((tuple(blocks), kind), f"resnet{blocks}")
        return f"smp Unet({name}), вход {self.in_channels} каналов, выход {self.classes}"

    def logits(self, x):
        """Как smp.Unet.forward: вход кратен 32."""
        return self.segmentation_head(self.decoder(self.encoder(x)))

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        y = self.logits(x)[..., :H, :W]
        out = {"sem": y[:, :N_CLASSES]}
        if self.classes == N_CLASSES + 5:
            r = y[:, N_CLASSES:]
            out.update(dist=torch.sigmoid(r[:, :1]), orient=F.normalize(r[:, 1:3], dim=1, eps=1e-6),
                       heights=torch.sigmoid(r[:, 3:5]))
        return out
