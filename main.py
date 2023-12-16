import torch
from Net import Net
from NetTrainer import train_net
from Trajectories import get_trajectories
from SudokuGame import SudokuGame
import wandb
import torch.multiprocessing
import numpy as np

args = {
    'numIters': 50,
    'numMCTSSims': 200,  # Number of games moves for MCTS to simulate.
    'cpuct': 1,  # UCB hyperparameter
    'numGames': 500,  # number of games to play in one iteration
    'lr': 0.1,
    'epochs': 50,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'wandb': True,
    'board_size': 4,
    'augment': True,
    'temp': 0.8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'conf_offset': 0.5,
    'start_blanks': 9,
    'end_blanks': 12,
}


def main(filename=None):
    if args['wandb']:
        logger = wandb.init(project="alphazero_sudoku")
    else:
        logger = None
    g = SudokuGame(4)
    net = Net(game=g)
    if filename is not None:
        net.load_state_dict(torch.load(filename))
    net.to(args['device'])
    iters_so_far = 0
    num_blanks = args['start_blanks']
    for i in range(args['numIters']):
        data, percent_perfect_games, avg_score = get_trajectories(net, args, logger, num_blanks)
        logger.log({"num_blanks": num_blanks})
        train_net(net, data, args, logger)
        iters_so_far += 1
        if (percent_perfect_games > 0.985 and iters_so_far >= 5) or (iters_so_far >= 20):
            torch.save(net.state_dict(), 'checkpoints/post_' + str(num_blanks) + '.pth')
            num_blanks += 1
            iters_so_far = 0
            if (num_blanks > args['end_blanks']): break


    torch.save(net.state_dict(), 'checkpoints/checkpoint_final.pth')


if __name__ == "__main__":
    main(filename=r'C:\Users\Danny Han\Desktop\alphazero-sudoku\checkpoints\run_8_blanks_2\checkpoint_20.pth')
