import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from SudokuNNet import SudokuNNet as onnet
from sklearn.model_selection import train_test_split

args = dotdict({
    'lr': 0.2,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples, wandb_logger):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Split the examples into training and validation sets
        train_examples, val_examples = train_test_split(examples, test_size=0.2)

        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            # Training phase
            batch_count = int(len(train_examples) / args.batch_size)
            t = tqdm(range(batch_count), desc='Training Net', position=0, leave=True)
            for _ in t:
                sample_ids = np.random.randint(len(train_examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[train_examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                wandb_logger.log({"Loss_pi": pi_losses.avg, "Loss_v": v_losses.avg})

            # Validation phase
            self.nnet.eval()
            with torch.no_grad():
                val_pi_losses = AverageMeter()
                val_v_losses = AverageMeter()
                max_policy_elements = []

                for board, pi, v in val_examples:
                    board = torch.FloatTensor(board.astype(np.float64)).unsqueeze(0)
                    target_pi = torch.FloatTensor(np.array(pi)).unsqueeze(0)
                    target_v = torch.FloatTensor(np.array(v).astype(np.float64)).unsqueeze(0)

                    if args.cuda:
                        board, target_pi, target_v = board.cuda(), target_pi.cuda(), target_v.cuda()

                    out_pi, out_v = self.nnet(board)
                    l_pi = self.loss_pi(target_pi, out_pi)
                    l_v = self.loss_v(target_v, out_v)

                    val_pi_losses.update(l_pi.item(), board.size(0))
                    val_v_losses.update(l_v.item(), board.size(0))

                    max_policy_elements.append(torch.max(torch.exp(out_pi)).item())

                # Update tqdm with validation losses
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg, Val_loss_pi=val_pi_losses.avg,
                              Val_loss_v=val_v_losses.avg)

                # Calculate and print statistics for policy's max element
                max_policy_mean = np.mean(max_policy_elements)
                max_policy_std = np.std(max_policy_elements)
                print(f"Validation - Mean of max policy element: {max_policy_mean:.4f}, Std Dev: {max_policy_std:.4f}")

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(-1, self.board_x + 1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
