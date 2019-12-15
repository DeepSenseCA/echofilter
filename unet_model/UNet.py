import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision


class UNet(nn.Module):

    __down = []
    __up = []

    def __init__(self, layers=4, input_channel=1):
        super(UNet, self).__init__()

        kernel_size = 3
        up_kernel_size = 2
        down_output_channel = 64
        last_layer_output_channel = 2
        last_layer_kernel = 1

        for i in range(0, layers+1):
            self.__down_init__(input_channel, down_output_channel, kernel_size)
            input_channel = down_output_channel
            down_output_channel = down_output_channel * 2

        up_output_channel = input_channel // 2
        # input_channel = down_output_channel

        for i in range(1, layers + 1):
            self.__up_init__(input_channel, up_output_channel, up_kernel_size, kernel_size)
            input_channel = up_output_channel
            up_output_channel = up_output_channel // 2

        # input_channel = up_output_channel
        self.__up.append(nn.Conv2d(input_channel, last_layer_output_channel, last_layer_kernel))

    def __down_init__(self, input_channel, output_channel, kernel_size):
        self.__down.append(nn.Conv2d(input_channel, output_channel, kernel_size))
        self.__down.append(nn.Conv2d(output_channel, output_channel, kernel_size))

    def __up_init__(self, input_channel, output_channel, up_kerner_size=2, kernel_size=3, scale_factor=2, padding_size=(1, 0, 1, 0)):
        self.__up.append(nn.Upsample(scale_factor=scale_factor))
        self.__up.append(torch.nn.ZeroPad2d(padding_size))
        self.__up.append(nn.Conv2d(input_channel, output_channel, up_kerner_size))
        self.__up.append(nn.Conv2d(input_channel, output_channel, kernel_size))
        self.__up.append(nn.Conv2d(output_channel, output_channel, kernel_size))

    def makeMiddleConnection(self, x_down, x_up):
        start_x = (x_down.shape[2] - x_up.shape[2]) // 2
        end_x = x_down.shape[2] - start_x
        start_y = (x_down.shape[3] - x_up.shape[3]) // 2
        end_y = x_down.shape[3] - start_y
        x_down = x_down[:, :, start_x: end_x, start_y: end_y]
        return torch.cat((x_up, x_down), dim=1)

    def forward(self, x):
        middleConnections = []
        for i in range(0, len(self.__down), 2):
            x = f.relu(self.__down[i](x))
            x = f.relu(self.__down[i+1](x))
            middleConnections.append(x)
            x = f.max_pool2d(x, 2)

        j = len(middleConnections) - 1
        for i in range(0, len(self.__up) - 1, 5):
            x = self.__up[i](x)
            x = self.__up[i+1](x)
            x = f.relu(self.__up[i+2](x))
            x = self.makeMiddleConnection(middleConnections[j - 1], x)
            j = j - 1
            x = f.relu(self.__up[i+3](x))
            x = f.relu(self.__up[i+4](x))

        x = f.relu(self.__up[-1](x))
        return x


net = UNet()
input = torch.randn(1, 1, 572, 572)
out = net(input)
print(out)
