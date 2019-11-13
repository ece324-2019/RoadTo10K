import torch.nn as nn
import torch


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):

        super(MultiLayerPerceptron, self).__init__()

        ######

        # 4.3 YOUR CODE HERE
        self.input_size = input_size
        self.output_size = 1
        self.fc1 = nn.Linear(103, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

        ######

    def forward(self, features):

        ######

        # 4.3 YOUR CODE HERE
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

        ######
