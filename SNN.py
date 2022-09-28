import functools
from turtle import forward
import torch.nn as nn
import torch
import time

def rand(shape, low, high):
    """Tensor of random numbers, uniformly distributed on [low, high]."""
    return torch.rand(shape) * (high - low) + low

class SNN(nn.Module):

    def __init__(self, pd_rho, q1, dh_rho, q2, fc_network, n):
        """
            Neural Network for classification of point clouds represented by is distributed homology,
            which again is represented by persistence diagrams.

        Args:
            pd_rho (nn.Module):     Network for vectorizing persistence diagrams.
            pd_w (nn.Module):       Network for weighting of vectorized persistence diagrams.
            pd_q (int):             Dimension of persistence diagram vectors.
            dh_rho (nn.Module):     Network for vectorizing sets of vectorized persistence diagrams -> Vectorizing DH.
            dh_w (nn.Module):       Network for weighting vectorized DH.
            dh_q (int):             Dimension of the DH vectors.
            fc_network (nn.Module): Network for classifying DH vectors.
        """

        super(SNN, self).__init__()

        # All parts of the network for vectorizing each persistence diagram
        self.pd_rho = pd_rho
        #self.pd_w = pd_w
        self.q1 = q1

        # All parts of the network for vectorizing the set of vectorized persistence diagram
        self.dh_rho = dh_rho
        #self.dh_w = dh_w
        self.q2 = q2

        self.fc = fc_network

        self.n = n

    def forward(self, x): 
        
        #print('Input: ', x)
        X = []
        #X = torch.ones(size=(len(x), self.n, self.q1))# --> Creating a big m x n x q1 array, batch_size, num_pd, q1
        for i, sample in enumerate(x):
            dh = []
            for j, pd in enumerate(sample):
                dh.append(self.pd_rho(pd))
            X.append(torch.stack(dh))

        x = torch.stack(X)
        x = self.dh_rho(x) # --> takes m x n x q1, gives m x  q2
        x = self.fc(x)
        return x

class DeepSetOperator(nn.Module):

    def __init__(self, operator:list, dim:int) -> None:
        super().__init__()

        self.operator = set()
        for op in operator:
            if type(op) == int:
                if op > 0:
                    self.operator.add(functools.partial(torch.topk, k = op, largest = True))
                elif op < 0:
                    self.operator.add(functools.partial(torch.topk, k = -op, largest = False))
            elif op == 'max':
                self.operator.add(torch.amax)
            elif op == 'mean':
                self.operator.add(torch.mean)
            elif op == 'min':
                self.operator.add(torch.amin)
            elif op == '75':
                self.operator.add(functools.partial(torch.quantile, q=0.75))
            elif op == '25':
                self.operator.add(functools.partial(torch.quantile, q=0.25))
            elif op == 'median':
                self.operator.add(torch.median)
            elif op == 'var':
                self.operator.add(torch.var)

        lim = (dim)**-0.5 / 2
        self.weight = torch.nn.Parameter(data=rand(dim, -lim, lim))
        self.bias = torch.nn.Parameter(data=rand(dim, -lim, lim))

    def forward(self, x):
        X = []
        for op in self.operator:
            if op == torch.median:
                X.append(op(x * self.weight + self.bias, dim=-2)[0])
            elif type(op) == type(functools.partial(torch.topk)) and op.func == torch.topk:
                X.extend([i for i in op(x * self.weight + self.bias, dim=-2).values])
            else:
                X.append(op(x * self.weight + self.bias, dim=-2))

        X = torch.cat(X, dim = -1)
        return X

class TransposeLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.swapaxes(x, -1, -2)

class DeepSetLayer(nn.Module):

    def __init__(self, input_dim:int, output_dim:int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialisation tactic copied from nn.Linear in PyTorch
        lim = (self.input_dim)**-0.5 / 2

        # Alpha corresponds to the identity, beta to the all-ones matrix, and gamma to the additive bias.
        self.alpha = torch.nn.Parameter(data=rand((self.output_dim, self.input_dim), -lim, lim))
        self.beta = torch.nn.Parameter(data=rand((self.output_dim, self.input_dim), -lim, lim))
        self.gamma = torch.nn.Parameter(data=rand((self.output_dim), -lim, lim))

    def forward(self, x):
        # x has shape (batch, in_blocks, n)
        x = torch.swapaxes(x, -1, -2)
        x = torch.einsum('...jz, ij -> ...iz', x, self.alpha) + torch.einsum('...jz, ij -> ...iz', x.sum(axis=-1)[..., None], self.beta) + self.gamma[..., None]
        x = torch.swapaxes(x, -1, -2)
        return x