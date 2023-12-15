import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, game):
        super(Net, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Expanded network layers
        self.fc1 = nn.Linear(self.board_x * self.board_y * (self.board_x + 1), 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, self.action_size)
        self.fc6 = nn.Linear(256, 1)  # Value head

    def forward(self, s):
        # Flatten the input
        s = s.view(s.size(0), -1)

        # Forward pass through the network with residual connections
        x = F.relu(self.bn1(self.fc1(s)))
        identity = x  # Save for the residual connection

        # Adding residual connection from fc1 to fc2
        x = F.relu(self.bn2(self.fc2(x))) + identity

        identity = x  # Update identity for the next residual connection

        # Adding residual connection from fc2 to fc3
        x = F.relu(self.bn3(self.fc3(x))) + identity

        x = F.relu(self.bn4(self.fc4(x)))  # No residual connection here
        logits = self.fc5(x)
        value = self.fc6(x)

        return F.log_softmax(logits, dim=1), F.sigmoid(value)
