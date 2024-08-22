import torch
import torch.nn as nn

from basic_mdl import basic_mdl

tanh_ = nn.Tanh()

def final_activation(env, a):
    return tanh_(a) * env.act_rng

class Policy(nn.Module):
    def __init__(self, env, nl=2, nn=200, act='relu'):
        super().__init__()
        self.env = env
        self.act = act
        self._g = basic_mdl(env.n, env.m, n_hid_layers=nl, act=act, n_hidden=nn, dropout=0.0)
        self.reset_parameters()

    def reset_parameters(self, w=1.0):
        self._g.reset_parameters(w)
    
    def forward(self, s, t):
        s = s.to(self.env.device)  
        a = self._g(s)
        return final_activation(self.env, a)