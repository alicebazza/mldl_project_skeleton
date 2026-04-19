import torch
import torch.nn as nn
import torch.nn.functional as F

class FC2Layer(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_hidden: int, output_size: int
    ) -> None:
        """
        Simple MLP model

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_hidden: size of the hidden dimension to use
        :param output_size: expected size of the output (e.g. number of classes if you are in a classification task)
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size * input_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]

        :returns: predictions with size [batch, output_size]
        """
        x = x.view(x.shape[0], -1)
        o = self.network(x)
        return o


class CNN(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_feature: int, output_size: int
    ) -> None:
        """
        Simple model that uses 3x3 convolutions

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_feature: size of the hidden dimensions to use (i.e. output channels for the conv layers)
        :param output_size: expected size of the output
        """
        super().__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=n_feature, kernel_size=3
        )
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=3)
        self.fc1 = nn.Linear(n_feature * 5 * 5, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self,
                x: torch.Tensor,
                return_conv1: bool = False,
                return_conv2: bool = False,
                return_conv3: bool = False
        ) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        :param return_conv1: if True return the feature maps of the first convolution
        :param return_conv2: if True return the feature maps of the second convolution
        :param return_conv3: if True return the feature maps of the third convolution

        :returns: predictions with size [batch, output_size]
        """
        x = self.conv1(x)
        if return_conv1:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        if return_conv2:
            return x
        x = F.relu(x)

        x = self.conv3(x)
        if return_conv3:
            return x
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

