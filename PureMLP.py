import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from SudokuGame import SudokuGame as Game
from Play import Play, string_2_array
import numpy as np
import pandas as pd
import wandb

import torch.nn as nn
import torch.nn.functional as F

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
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, self.action_size)
        self.fc6 = nn.Linear(256, 1) # Value head

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

        return F.log_softmax(logits, dim=1), value



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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model, num_epochs=10):
    wandb.init(project="alphazero_sudoku")
    model.to(device)
    g = Game(4)
    df = pd.read_csv('sudoku-4.csv')
    #hold out 100 samples for validation
    df_train = df.sample(frac=1)
    df_val = df_train[:100]
    df_train = df_train[100:]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    model.train()
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
            # print(puzzles.shape)
            # # print some two dim puzzles and corresponding targets
            # print(Game.three_dim_to_two_dim(puzzles[0].cpu().numpy()))
            # print(target[0])
            # print(Game.three_dim_to_two_dim(puzzles[1].cpu().numpy()))
            # print(target[1])
            # assert False

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

            for i in range(num_epochs):
                output, _ = model(puzzles)
                print(output.shape)
                print(target.shape)
                loss = loss_pi(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                output_max = torch.max(torch.exp(output))
                # Update the progress bar description
                pbar.set_description(f"Training (Loss: {loss.item():.4f}, output_max: {output_max.item()})")
                pbar.update(1)
                wandb.log({"Loss": loss.item()})
            model.eval()
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
                output, _ = model(puzzles)

                numpy_targ = np.array([Game.two_dim_to_three_dim(solutions[i])[1:, :, :] for i in range(batch_size)])
                target = torch.tensor(numpy_targ, dtype=torch.float32).to(device)
                target = target.view(target.size(0), -1)
                target = target - puzzles[:, 1:, :, :].view(target.size(0), -1)
                loss = loss_pi(output, target)
                wandb.log({"Val Loss": loss.item()})
                model.train()
    print(losses[0], losses[-1])

if __name__ == '__main__':
    g = Game(4)
    model = SudokuNN(g)
    model.to(device)
    main(model)