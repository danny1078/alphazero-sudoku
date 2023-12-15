import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

# from SudokuNNet import SudokuNNet as onnet
from PureMLP import SudokuNN as onnet
from sklearn.model_selection import train_test_split
from SudokuGame import SudokuGame as Game
from Play import Play, string_2_array
import random
import pandas as pd

args = dotdict({
    'lr': 0.2,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game)#, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train_old(self, examples, wandb_logger):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Split the examples into training and validation sets
        train_examples, val_examples = train_test_split(examples, test_size=0.2)
        train_boards = [x[0] for x in train_examples]
        train_pis = [x[1] for x in train_examples]
        train_vs = [x[2] for x in train_examples]
        val_boards = [x[0] for x in val_examples]
        val_pis = [x[1] for x in val_examples]
        val_vs = [x[2] for x in val_examples]
        # print(train_boards[0].shape)
        # print(SudokuGame.three_dim_to_two_dim(train_boards[0]))
        # s = SudokuGame(n=4)
        # action = np.random.choice(len(train_pis[0]), p=train_pis[0])
        # print(action)
        # next_board = s.getNextState(SudokuGame.three_dim_to_two_dim(train_boards[0]), action)
        # print(next_board)

        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        t = tqdm(range(args.epochs), desc='Training Net', position=0, leave=True)
        for epoch in t:
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            # Training phase
            batch_count = int(len(train_examples) / args.batch_size)
            for i in range(batch_count):
                boards = torch.FloatTensor(np.array(train_boards).astype(np.float64))[i * args.batch_size: (i + 1) * args.batch_size]
                target_pis = torch.FloatTensor(np.array(train_pis))[i * args.batch_size: (i + 1) * args.batch_size]
                target_vs = torch.FloatTensor(np.array(train_vs).astype(np.float64))[i * args.batch_size: (i + 1) * args.batch_size]

                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                # print dimensions of target and out pis
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + 1e2 * l_v
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
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
                max_policy = np.max(max_policy_elements)
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg, Val_loss_pi=val_pi_losses.avg,
                              Val_loss_v=val_v_losses.avg, Max_policy=max_policy)

    def predict(self, board, model):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(-1, self.board_x + 1, self.board_x, self.board_y)
        model.eval()
        with torch.no_grad():
            pi, v = model(board)

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


    def train(self, model, data, wandb_logger):
        def generatePuzzle(solution, num_blanks):
            # randomly mask num_blanks squares in solution with zeros
            # return solution and masked solution
            # solution is a 9x9 numpy array
            # masked_solution is a 9x9 numpy array
            gameN = 4
            masked_solution = solution.copy()
            mask = np.random.choice((gameN ** 2), num_blanks, replace=False)
            for i in mask:
                masked_solution[i // gameN][i % gameN] = 0
            return masked_solution

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(device)
        g = Game(4)
        df = pd.read_csv('../sudoku-4.csv')
        #hold out 100 samples for validation
        df_train = df.sample(frac=1)
        df_val = df_train[:100]
        df_train = df_train[100:]

        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=0.2)
        self.nnet.train()
        num_iters = 1000
        batch_size = 256
        losses = []
        with tqdm(total=num_iters, desc="Training") as pbar:  # Wrap the loop with tqdm
            for _ in range(num_iters):

                solutions = []
                for _ in range(batch_size):
                    solutions.append(string_2_array(df_train.sample(1)['solution'].values[0]))
                puzzles = []
                for i in range(batch_size):
                    twodim = generatePuzzle(solutions[i], 1)
                    threedim = Game.two_dim_to_three_dim(twodim)
                    puzzles.append(threedim)

                puzzles = torch.tensor(np.array(puzzles), dtype=torch.float32).to(device)
                numpy_targ = np.array([Game.two_dim_to_three_dim(solutions[i])[1:, :, :] for i in range(batch_size)])
                target = torch.tensor(numpy_targ, dtype=torch.float32).to(device)
                target = target.view(target.size(0), -1)
                target = target - puzzles[:, 1:, :, :].view(target.size(0), -1)

                # Step 1: Generate noise
                noise = torch.rand(target.size(), device=device) * 0.01  # Small positive noise

                # Step 2: Normalize the noise for each batch
                noise_sum = noise.sum(dim=1, keepdim=True)
                epsilon = 0.01  # Small value
                noise = noise / noise_sum * epsilon

                # Step 3: Add noise to target
                target = target + noise

                # Step 4: Normalize target so that the sum of each batch is 1
                target_sum = target.sum(dim=1, keepdim=True)
                target = target / target_sum

                for i in range(10):
                    output, _ = self.nnet(puzzles)
                    loss = self.loss_pi(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    output_max = torch.max(torch.exp(output))
                    # Update the progress bar description
                    pbar.set_description(f"Training (Loss: {loss.item():.4f}, output_max: {output_max.item()})")
                    pbar.update(1)
                    wandb_logger.log({"Loss": loss.item()})
                self.nnet.eval()
                with torch.no_grad():
                    solutions = []
                    for _ in range(batch_size):
                        solutions.append(string_2_array(df_val.sample(1)['solution'].values[0]))
                    puzzles = []
                    for i in range(batch_size):
                        twodim = generatePuzzle(solutions[i], 1)
                        threedim = Game.two_dim_to_three_dim(twodim)
                        puzzles.append(threedim)

                    puzzles = torch.tensor(np.array(puzzles), dtype=torch.float32).to(device)
                    output, _ = self.nnet(puzzles)

                    numpy_targ = np.array([Game.two_dim_to_three_dim(solutions[i])[1:, :, :] for i in range(batch_size)])
                    target = torch.tensor(numpy_targ, dtype=torch.float32).to(device)
                    target = target.view(target.size(0), -1)
                    target = target - puzzles[:, 1:, :, :].view(target.size(0), -1)
                    loss = self.loss_pi(output, target)
                    wandb_logger.log({"Val Loss": loss.item()})
                    self.nnet.train()
        print(losses[0], losses[-1])
