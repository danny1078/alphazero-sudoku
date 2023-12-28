import torch
from Net import Net, AttentionNet
from NetTrainer import train_net
from Trajectories import get_trajectories, eval_trajectories
from SudokuGame import SudokuGame
import wandb
import torch.multiprocessing
import numpy as np
import pandas as pd

args = {
    'maxNumIters': 20,
    'numMCTSSims': 200,  # Number of games moves for MCTS to simulate.
    'cpuct': 1,  # UCB hyperparameter
    'numGames': 500,  # number of games to play in one iteration
    'lr': 0.2,
    'epochs': 30,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'wandb': True,
    'board_size': 4,
    'augment': True,
    'temp': 0.8,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'conf_offset': 0.5,
    'start_blanks': 2,
    'end_blanks': 12,
}


def main(filename=None):
    if args['wandb']:
        logger = wandb.init(project="alphazero_sudoku")
    else:
        logger = None
    g = SudokuGame(args['board_size'])
    net = AttentionNet(game=g)
    if filename is not None:
        net.load_state_dict(torch.load(filename))
    net.to(args['device'])
    iters_so_far = 0
    num_blanks = args['start_blanks']
    while True:
        data, percent_perfect_games, avg_score = get_trajectories(net, args, logger, num_blanks)
        if logger is not None:
            logger.log({"num_blanks": num_blanks})
        train_net(net, data, args, logger)
        iters_so_far += 1
        if (percent_perfect_games > 0.98 and iters_so_far >= 10) or (iters_so_far >= args['maxNumIters']):
            torch.save(net.state_dict(), 'checkpoints/post_' + str(num_blanks) + '.pth')
            num_blanks += 1
            iters_so_far = 0
            if (num_blanks > args['end_blanks']): break


    torch.save(net.state_dict(), 'checkpoints/checkpoint_final.pth')


def evaluate_main(filename):
    g = SudokuGame(args['board_size'])
    net = Net(game=g)
    net.load_state_dict(torch.load(filename))
    net.to(args['device'])
    num_mcts_sims = [5 * i for i in range(1, 21)]
    percents = []
    avg_scores = []
    for num_mcts_sim in num_mcts_sims:
        args['numMCTSSims'] = num_mcts_sim
        data, percent_perfect_games, avg_score = eval_trajectories(net, args)
        percents.append(percent_perfect_games)
        avg_scores.append(avg_score)
    df = pd.DataFrame({'num_mcts_sims': num_mcts_sims, 'percent_perfect_games': percents, 'avg_scores': avg_scores})
    df.to_csv('num_mcts_sims.csv')


if __name__ == "__main__":
    main()
