import torch
import torch.nn as nn


class ImageAugmentation(nn.Module):
    def __init__(self, zero_prob=0.1, noise_prob=0.1):
        """
        初始化数据增强模块。

        参数:
        - zero_prob (float): 将输入置为0的概率。
        - noise_prob (float): 将输入替换为随机噪声的概率。
        """
        super(ImageAugmentation, self).__init__()
        self.zero_prob = zero_prob
        self.noise_prob = noise_prob

    def forward(self, x):
        """
        对输入进行数据增强，输出形状与输入相同。

        参数:
        - x (torch.Tensor): 输入张量，任意形状。

        返回:
        - torch.Tensor: 增强后的输出张量，形状与输入相同。
        """
        # 如果不是 training 模式，则直接返回输入
        if not self.training:
            return x

        if torch.rand(1).item() < self.zero_prob:
            # 以 zero_prob 的概率将输入置为 0
            return torch.zeros_like(x)

        if torch.rand(1).item() < self.noise_prob:
            # 以 noise_prob 的概率将输入替换为随机噪声
            return torch.randn_like(x)

        # 默认返回原始输入
        return x
