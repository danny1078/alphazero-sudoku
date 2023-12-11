import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SudokuNNet(nn.Module):
    def __init__(self, game, args):
        # game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(SudokuNNet, self).__init__()
        # Convolution layers
        self.conv_3_3 = nn.Conv2d(self.board_x + 1, args.num_channels, 3, stride=3)
        self.conv_1_9 = nn.Conv2d(self.board_x + 1, args.num_channels, (1, 9), stride=1)
        self.conv_9_1 = nn.Conv2d(self.board_x + 1, args.num_channels, (9, 1), stride=1)

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

        return F.log_softmax(policy, dim=1), torch.sigmoid(value)
