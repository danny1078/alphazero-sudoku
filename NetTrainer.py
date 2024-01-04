import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_net(net, data, args, logger):
    train_loader, val_loader = makeDataLoader(data, args)
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    t = tqdm(range(args['epochs']), desc='Training Net', position=0, leave=True)

    for epoch in t:
        net.train()

        # Training phase
        pi_losses = []
        v_losses = []
        for boards, target_pis, target_vs in train_loader:
            out_pi, out_v = net(boards)
            lambda_entropy = 0.3
            l_pi = loss_pi_with_entropy(target_pis, out_pi, lambda_entropy)
            l_v = loss_v(target_vs, out_v)
            total_loss = l_pi + 1e2 * l_v
            pi_losses.append(l_pi.item())
            v_losses.append(l_v.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if logger is not None:
                logger.log({"Loss_pi": pi_losses[-1], "Loss_v": v_losses[-1]})

        # Validation phase
        net.eval()
        val_pi_losses = []
        val_v_losses = []
        max_policy_elements = []

        with torch.no_grad():
            for boards, target_pis, target_vs in val_loader:
                out_pi, out_v = net(boards)
                l_pi = loss_pi(target_pis, out_pi)
                l_v = loss_v(target_vs, out_v)

                val_pi_losses.append(l_pi.item())
                val_v_losses.append(l_v.item())
                max_policy_elements.append(torch.max(torch.exp(out_pi)).item())

            if logger is not None:
                logger.log({"Val_loss_pi": np.array(val_pi_losses).mean(), "Val_loss_v": np.array(val_v_losses).mean()})

        # Update tqdm with validation losses
        max_policy = np.max(max_policy_elements)
        t.set_postfix(Loss_pi=np.array(pi_losses).mean(),
                      Loss_v=np.array(v_losses).mean(),
                      Val_loss_pi=np.array(val_pi_losses).mean(),
                      Val_loss_v=np.array(val_v_losses).mean(),
                      Max_policy=max_policy)

        scheduler.step(np.array(val_pi_losses).mean() + np.array(val_v_losses).mean())



def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]

def loss_pi_with_entropy(targets, log_outputs, lambda_entropy):
    """
    Calculate the policy loss with an entropy penalty, using log-softmax outputs.

    :param targets: The target policy distributions.
    :param log_outputs: The predicted log-probabilities (log-softmax outputs) from the network.
    :param lambda_entropy: Scaling factor for the entropy penalty.
    :param epsilon: A small constant to improve numerical stability.
    :return: Combined loss value.
    """
    # Cross-entropy loss (since outputs are log-probabilities)
    cross_entropy_loss = -torch.sum(targets * log_outputs, dim=1).mean()

    # Convert log-probabilities to probabilities for entropy calculation
    probabilities = torch.exp(log_outputs)

    # Entropy calculation with epsilon to avoid log(0)
    entropy = -torch.sum(probabilities * log_outputs, dim=1).mean()
    entropy_penalty = lambda_entropy * entropy

    # Cap the entropy penalty at 1 and ensure it's not negative
    entropy_penalty = torch.clamp(entropy_penalty, min=0, max=1.0)

    return cross_entropy_loss + entropy_penalty


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

class BoardDataset(torch.utils.data.Dataset):
    def __init__(self, boards, pis, vs, cuda=False):
        # Convert entire lists to numpy arrays
        boards_np = np.array(boards, dtype=np.float32)
        pis_np = np.array(pis, dtype=np.float32)
        vs_np = np.array(vs, dtype=np.float32)

        # Convert numpy arrays to PyTorch tensors
        self.boards = torch.from_numpy(boards_np).contiguous()
        self.pis = torch.from_numpy(pis_np).contiguous()
        self.vs = torch.from_numpy(vs_np).contiguous()

        # Move tensors to GPU if available
        if cuda:
            self.boards = self.boards.cuda()
            self.pis = self.pis.cuda()
            self.vs = self.vs.cuda()


    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.pis[idx], self.vs[idx]


def makeDataLoader(data, args):
    train_examples, val_examples = train_test_split(data, test_size=0.2)

    # Unpack and separate the data
    train_boards, train_pis, train_vs = zip(*train_examples)
    val_boards, val_pis, val_vs = zip(*val_examples)

    # Create dataset instances
    train_dataset = BoardDataset(train_boards, train_pis, train_vs, cuda=args['cuda'])
    val_dataset = BoardDataset(val_boards, val_pis, val_vs, cuda=args['cuda'])

    # DataLoader setup
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return train_loader, val_loader

