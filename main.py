from Train import Train
from SudokuGame import SudokuGame as Game
from NNet import NNetWrapper as nn
from utils import *
from PureMLP import SudokuNN


args = dotdict({
    'numIters': 100,
    'numEps': 20,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'cpuct': 1.41,                 # UCB hyperparameter

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'numGames': 500 # number of games to play in one iteration

})


def main():
    g = Game(4) # size of board

    nnet = nn(g)
    model = SudokuNN(g)

    if args.load_model: # for loading a partially trained model
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    t = Train(g, nnet, args)

    print('Starting the learning process 🎉')
    t.learn(model)


if __name__ == "__main__":
    main()
