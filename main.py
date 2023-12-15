import torch
from Net import Net
from NetTrainer import train_net
from Trajectories import get_trajectories
from SudokuGame import SudokuGame
import wandb

args = {
    'numIters': 50,
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,                 # UCB hyperparameter
    'numGames': 500, # number of games to play in one iteration
    'lr': 0.2,
    'epochs': 15,
    'batch_size': 128,
    'num_workers': 0,
    'cuda': torch.cuda.is_available(),
    'wandb': True,
    'board_size': 4,
    'augment': False,
    'temp': 1,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

}


def main():
    if args['wandb']:
        logger = wandb.init(project="alphazero_sudoku")
    else:
        logger = None
    g = SudokuGame(4)
    net = Net(game=g)
    net.to(args['device'])

    for _ in range(args['numIters']):
        data = get_trajectories(net, args, logger)
        train_net(net, data, args, logger)


if __name__ == "__main__":
    main()
