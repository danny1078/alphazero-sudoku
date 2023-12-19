import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def predict(self, s):
        """
        Input:
            s: a batch of boards
        Returns:
            pi: a batch of action probabilities
            v: a batch of value predictions
        """

        board = torch.FloatTensor(s.astype(np.float64))
        if torch.cuda.is_available():
            board = board.contiguous().cuda()
        board = torch.unsqueeze(board, 0)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]


class AttentionNet(nn.Module):
    def __init__(self, game):
        super(AttentionNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Define the size of the input features
        input_features = self.board_x * self.board_y * (self.board_x + 1)

        # Expanded network layers
        self.fc1 = nn.Linear(input_features, 512)
        self.bn1 = nn.BatchNorm1d(512)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, 512))

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

        # Further network layers
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, self.action_size)
        self.fc6 = nn.Linear(512, 1)  # Value head

    def forward(self, s):
        # Flatten the input
        s = s.view(s.size(0), -1)
        s = F.relu(self.bn1(self.fc1(s)))

        # Apply positional encoding
        s = s.unsqueeze(1) + self.positional_encoding  # Reshape to (batch_size, 1, 256)

        # Self-attention
        s = s.transpose(0, 1)  # Reshape to (1, batch_size, 256)
        s, _ = self.self_attention(s, s, s)
        s = s.transpose(0, 1)  # Reshape back to (batch_size, 1, 256)
        s = s.squeeze(1)

        # Adding residual connection from fc1 to fc2
        identity = s
        s = F.relu(self.bn2(self.fc2(s))) + identity

        # Further layers
        s = F.relu(self.bn3(self.fc3(s)))
        s = F.relu(self.bn4(self.fc4(s)))

        logits = self.fc5(s)
        value = self.fc6(s)

        return F.log_softmax(logits, dim=1), torch.sigmoid(value)


    def predict(self, s):
        """
        Input:
            s: a batch of boards
        Returns:
            pi: a batch of action probabilities
            v: a batch of value predictions
        """

        board = torch.FloatTensor(s.astype(np.float64))
        board = torch.unsqueeze(board, 0)
        if torch.cuda.is_available():
            board = board.contiguous().cuda()
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

