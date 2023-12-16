import torch
from Net import Net
from NetTrainer import train_net
from Trajectories import get_trajectories
from SudokuGame import SudokuGame
import wandb
import torch.multiprocessing
import numpy as np

args = {
    'numIters': 100,
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,                 # UCB hyperparameter
    'numGames': 500, # number of games to play in one iteration
    'lr': 0.2,
    'epochs': 15,
    'batch_size': 128,
    'num_workers': 1,
    'cuda': torch.cuda.is_available(),
    'wandb': True,
    'board_size': 4,
    'augment': False,
    'temp': 0.8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'conf_offset': 0.5,

}


def main():
    if args['wandb']:
        logger = wandb.init(project="alphazero_sudoku")
    else:
        logger = None
    g = SudokuGame(4)
    net = Net(game=g)
    net.to(args['device'])
    for i in range(args['numIters']):
        num_blanks = np.floor(i / args['numIters'] * 3) + 2
        data = get_trajectories(net, args, logger, num_blanks)
        logger.log({"num_blanks": num_blanks})
        train_net(net, data, args, logger)


if __name__ == "__main__":
    main()
