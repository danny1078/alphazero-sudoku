import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from SudokuGame import SudokuGame as Game
from Play import Play, string_2_array
import numpy as np
import pandas as pd

class SudokuNN(nn.Module):
    def __init__(self, game):
        super(SudokuNN, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        # Expanded network layers
        self.fc1 = nn.Linear(self.board_x * self.board_y * (self.board_x + 1), 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, self.action_size)

    def forward(self, s):
        # Flatten the input
        s = s.view(s.size(0), -1)

        # Forward pass through the network
        x = F.relu(self.bn1(self.fc1(s)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        logits = self.fc5(x)

        return F.log_softmax(logits, dim=1)


## Training loop:

def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]

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

def main(model):
    g = Game(4)
    df = pd.read_csv('sudoku-4.csv')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    model.train()
    num_iters = 1000
    batch_size = 128
    losses = []

    with tqdm(total=num_iters, desc="Training") as pbar:  # Wrap the loop with tqdm
        for _ in range(num_iters):

            solutions = []
            for _ in range(batch_size):
                solutions.append(string_2_array(df.sample(1)['solution'].values[0]))
            puzzles = []
            for i in range(batch_size):
                twodim = generatePuzzle(solutions[i], 1)
                threedim = Game.two_dim_to_three_dim(twodim)
                puzzles.append(threedim)
            
            puzzles = np.array(puzzles)
            puzzles = torch.tensor(puzzles, dtype=torch.float32)
            output = model(puzzles)

            numpy_targ = np.array([Game.two_dim_to_three_dim(solutions[i])[1:, :, :] for i in range(batch_size)])

            target = torch.tensor(numpy_targ, dtype=torch.float32)
            target = target.view(target.size(0), -1)

            target = target - puzzles[:, 1:, :, :].view(target.size(0), -1)

            loss = loss_pi(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Update the progress bar description
            pbar.set_description(f"Training (Loss: {loss.item():.4f})")
            pbar.update(1)

    print(losses[0], losses[-1])

if __name__ == '__main__':
    g = Game(4)
    model = SudokuNN(g)
    main(model)