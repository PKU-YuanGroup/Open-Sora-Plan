import torch
import torch.nn as nn
import torch.nn.functional as F

# Used for image self-supervised learning
def nt_xent_loss(q,k,temp=1.0):
    assert q.shape==k.shape
    q = F.normalize(q, dim=-1, p=2)
    k = F.normalize(k, dim=-1, p=2)
    c=torch.cat([q,k],dim=0)
    N = q.shape[0]
    N2 = N*2
    sim = torch.einsum('xd,yd->xy', c, c)
    sim = sim / temp
    sim = sim[~torch.eye(N2, dtype=bool,device=c.device)].reshape(N2, N2-1)
    labels = torch.cat([torch.arange(N-1,N2-1,device=sim.device), torch.arange(N,device=sim.device)], dim=0)
    loss = F.cross_entropy(sim, labels, reduction='sum')
    return loss/N2

class ClipLoss(nn.Module):
    def __init__(self,temp):
        super(ClipLoss, self).__init__()
        self.temp=temp

    def forward(self,i,t):
        assert i.shape==t.shape
        i = F.normalize(i, dim=-1, p=2)
        t = F.normalize(t, dim=-1, p=2)
        sim = torch.einsum('xd,yd->xy', i, t)/self.temp
        sim = sim.exp()
        mask=torch.eye(sim.shape[0], dtype=bool,device=sim.device)
        i2t_pos=sim[mask]
        i2t_neg=sim[~mask].reshape(sim.shape[0],sim.shape[0]-1).sum(dim=-1)
        i2t_loss = -torch.log(i2t_pos/(i2t_neg)).mean()

        sim = sim.t()
        t2i_pos=sim[mask]
        t2i_neg=sim[~mask].reshape(sim.shape[0],sim.shape[0]-1).sum(dim=-1)
        t2i_loss = -torch.log(t2i_pos/(t2i_neg)).mean()

        return (i2t_loss+t2i_loss)/2