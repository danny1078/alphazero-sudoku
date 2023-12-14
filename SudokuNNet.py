import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from vit import VisionTransformer


class SudokuNNet(nn.Module):
    def __init__(self, game, args):
        super(SudokuNNet, self).__init__()
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
        self.policy_head = nn.Linear(256, self.action_size)
        self.value_head = nn.Linear(256, 1)

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
        logits = self.policy_head(x)
        value = self.value_head(x)

        return F.log_softmax(logits, dim=1), value

class SudokuNNet_old(nn.Module):
    def __init__(self, game, args):
        # game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.subgrid_size = int(self.board_x ** 0.5)

        super(SudokuNNet_old, self).__init__()
        # Convolution layers
        self.conv_3_3 = nn.Conv2d(self.board_x + 1, args.num_channels, self.subgrid_size, stride=self.subgrid_size)
        self.conv_1_9 = nn.Conv2d(self.board_x + 1, args.num_channels, (1, self.board_x), stride=1)
        self.conv_9_1 = nn.Conv2d(self.board_x + 1, args.num_channels, (self.board_x, 1), stride=1)

        # Batch normalization for convolution layers
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        # Fully connected layers for each convolution output
        fc_input_dim = args.num_channels * self.board_x
        self.fc1_3_3 = nn.Linear(fc_input_dim, 1024)
        self.fc1_1_9 = nn.Linear(fc_input_dim, 1024)
        self.fc1_9_1 = nn.Linear(fc_input_dim, 1024)

        # Concatenated fully connected layer
        self.fc2 = nn.Linear(1024 * 3, 1024)

        # Separate heads for policy and value
        self.policy_head = nn.Linear(1024, self.action_size)
        self.value_head = nn.Linear(1024, 1)

    def forward(self, s):
        s = s.view(-1, self.board_x + 1, self.board_x, self.board_y)

        # Apply convolutions and batch normalization
        conv_3_3_out = F.relu(self.bn1(self.conv_3_3(s)))
        conv_1_9_out = F.relu(self.bn2(self.conv_1_9(s)))
        conv_9_1_out = F.relu(self.bn3(self.conv_9_1(s)))

        # Flatten outputs for fully connected layers
        conv_3_3_out = conv_3_3_out.view(-1, self.args.num_channels * self.board_x)
        conv_1_9_out = conv_1_9_out.view(-1, self.args.num_channels * self.board_x)
        conv_9_1_out = conv_9_1_out.view(-1, self.args.num_channels * self.board_x)

        # Fully connected layers for each output
        fc1_3_3_out = F.relu(self.fc1_3_3(conv_3_3_out))
        fc1_1_9_out = F.relu(self.fc1_1_9(conv_1_9_out))
        fc1_9_1_out = F.relu(self.fc1_9_1(conv_9_1_out))

        # Concatenate and pass through final fully connected layer
        fc2_input = torch.cat((fc1_3_3_out, fc1_1_9_out, fc1_9_1_out), dim=1)
        fc2_out = F.relu(self.fc2(fc2_input))

        # Output heads for policy and value
        policy = self.policy_head(fc2_out)
        value = self.value_head(fc2_out)

        return F.log_softmax(policy, dim=1), torch.exp(3) * torch.relu(value)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SudokuNNet_vit(nn.Module):
    def __init__(self, game, args):
        super(SudokuNNet_vit, self).__init__()
        # Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.subgrid_size = int(self.board_x ** 0.5)
        # Initialize the Vision Transformer
        self.num_classes = 512
        self.vit = VisionTransformer(
            image_size=self.board_x,
            patch_size=self.subgrid_size,
            num_classes=self.num_classes,
            hidden_dim=self.board_x + 1,
            num_heads=5,
            mlp_dim=512,
            dropout=0,
            num_layers=8,
        )
        # Output heads for policy and value

        self.policy_head = nn.Linear(self.num_classes, 128)
        self.policy_head2 = nn.Linear(128, self.action_size)
        self.value_head = nn.Linear(self.num_classes, 128)
        self.value_head2 = nn.Linear(128, 1)

    def forward(self, s):
        # Pass the input through the Vision Transformer
        transformer_out = self.vit(s)

        # Output heads for policy and value
        policy = torch.relu(self.policy_head(transformer_out))
        policy = self.policy_head2(policy)
        value = torch.relu(self.value_head(transformer_out))
        value = self.value_head2(value)

        return F.log_softmax(policy, dim=1), torch.tanh(value)



