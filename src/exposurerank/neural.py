import torch
from torch import nn
from .factorization import MatrixFactorization

class NeuralInteraction(MatrixFactorization):
    def __init__(self,n_users,n_items,dimension,p0):
        super().__init__(n_users,n_items,dimension,p0)
        self.net=nn.Sequential(nn.Linear(3*dimension,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def interaction(self,p,q): return self.net(torch.cat((p,q,p*q),dim=1)).squeeze(1)
