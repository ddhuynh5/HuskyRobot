""" CNN for Husky Robot """

from torch import nn


class Network(nn.Module):
    """ Convolutional neural network base class """

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=3, stride=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=3)
        self.conv5 = nn.Conv2d(4, 2, kernel_size=1, stride=2)
        self.conv6 = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2, 6)

    def forward(self, x):
        """ Feed-Forward: Pass data into nn and perform calc """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.squeeze()

        return x