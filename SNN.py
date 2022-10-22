import functools
from turtle import forward
from unittest import result
from numpy import iterable
import torch.nn as nn
import torch
import time
from typing import List

def rand(shape, low, high):
    """Tensor of random numbers, uniformly distributed on [low, high]."""
    return torch.rand(shape) * (high - low) + low


#Vector = List[List[torch.tensor]]
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
class DSSN(nn.Module):

    def __init__(self, pd_rho, dh_rho, fc_network, device):

        super(DSSN, self).__init__()

        self.pers_lay1 = pd_rho
        self.pers_lay2 = dh_rho
        self.fc = fc_network

    def forward(self:nn.Module, x:List[List[torch.Tensor]]) -> torch.Tensor: 

        x = torch.stack([torch.stack([self.pers_lay1(pd) for pd in sample]) for sample in x])
        x = self.pers_lay2(x) # Batch_size x num_subsets x q1
        x = self.fc(x)

        return x

@torch.jit.ignore
class PersLay(nn.Module):

    @torch.jit.ignore
    def __init__(self, rho, phi, operator) -> None:
        super(PersLay, self).__init__()

        self.rho = rho      # Vectorization of elements, use DeepSet.Module
        self.phi = phi      # Weighting of elements, use DeepSet.Module
        self.op = operator

    @torch.jit.ignore
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Takes batch_size x input_dimension
        return self.op(self.rho(x)*self.phi(x))

class DeepSetOperator(nn.Module):
    @torch.jit.ignore
    def __init__(self, operator:list) -> None:
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
            elif op == 'sum':
                self.operator.add(torch.sum)

        '''lim = (dim)**-0.5 / 2
        self.weight = torch.nn.Parameter(data=rand(dim, -lim, lim))
        self.bias = torch.nn.Parameter(data=rand(dim, -lim, lim))'''

    @torch.jit.ignore
    def forward(self, x):
        X = []
        for op in self.operator:
            if op == torch.median:
                X.append(op(x, dim=-2)[0])
            elif type(op) == type(functools.partial(torch.topk)) and op.func == torch.topk:
                X.extend([i for i in op(x, dim=-2).values])
            else:
                X.append(op(x, dim=-2))

        if len(self.operator) > 0:
            X = torch.cat(X, dim = -1)
            return X
        else:
            return x

class TransposeLayer(nn.Module):
    @torch.jit.ignore
    def __init__(self) -> None:
        super().__init__()
    @torch.jit.ignore
    def forward(self, x):
        return torch.swapaxes(x, -1, -2)

class DeepSetLayer(nn.Module):
    @torch.jit.ignore
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
    @torch.jit.ignore
    def forward(self, x):
        # x has shape (batch, in_blocks, n)
        x = torch.swapaxes(x, -1, -2)
        x = torch.einsum('...jz, ij -> ...iz', x, self.alpha) + torch.einsum('...jz, ij -> ...iz', x.sum(axis=-1)[..., None], self.beta) + self.gamma[..., None]
        x = torch.swapaxes(x, -1, -2)
        return x
