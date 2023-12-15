import torch
from Net import Net
from NetTrainer import train_net
from Trajectories import get_trajectories
from SudokuGame import SudokuGame
import wandb

args = {
    'numIters': 100,
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'cpuct': 1.41,                 # UCB hyperparameter
    'numGames': 5000, # number of games to play in one iteration
    'lr': 0.2,
    'epochs': 100,
    'batch_size': 128,
    'num_workers': 32,
    'cuda': torch.cuda.is_available(),
    'wandb': True
}


def main():
    if args['wandb']:
        logger = wandb.init(project="alphazero_sudoku")
    else:
        logger = None
    g = SudokuGame(4)
    net = Net(game=g)

    for _ in range(args['numIters']):
        data = get_trajectories(net, args)
        train_net(net, data, args, logger)


if __name__ == "__main__":
    main()
