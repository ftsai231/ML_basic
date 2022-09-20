# https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.ReLu()
        self.fc2 = nn.ReLu()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x)
        x = F.relu(x)
        x = self.dropout2(x)

        output = F.log_softmax(x)

        return output


# Equates to one random 28x28 image
random_data = torch.rand((1, 1, 28, 28))
my_nn = Net()
result = my_nn(random_data)
print(result)