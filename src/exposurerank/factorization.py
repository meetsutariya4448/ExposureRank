import torch
from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self,n_users,n_items,dimension,p0):
        super().__init__()
        self.user=nn.Embedding(n_users,dimension);self.item=nn.Embedding(n_items,dimension)
        self.bu=nn.Embedding(n_users,1);self.ci=nn.Embedding(n_items,1)
        self.mu=nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(p0)))))
        nn.init.normal_(self.user.weight,0,.01);nn.init.normal_(self.item.weight,0,.01)
        nn.init.zeros_(self.bu.weight);nn.init.zeros_(self.ci.weight)
    def interaction(self,p,q): return (p*q).sum(dim=1)
    def forward(self,u,i): return self.mu+self.bu(u).squeeze(1)+self.ci(i).squeeze(1)+self.interaction(self.user(u),self.item(i))
    def penalty(self): return sum((p*p).sum() for p in (self.user.weight,self.item.weight,self.bu.weight,self.ci.weight))
