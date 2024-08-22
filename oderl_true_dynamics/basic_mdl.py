import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def get_act(act="relu"):
    if act=="relu":         return nn.ReLU()
    elif act=="elu":        return nn.ELU()
    elif act=="celu":       return nn.CELU()
    elif act=="leaky_relu": return nn.LeakyReLU()
    elif act=="sigmoid":    return nn.Sigmoid()
    elif act=="tanh":       return nn.Tanh()
    elif act=="sin":        return torch.sin
    elif act=="linear":     return nn.Identity()
    elif act=='softplus':   return nn.modules.activation.Softplus()
    elif act=='swish':      return lambda x: x*torch.sigmoid(x)
    else:                   return None

class basic_mdl(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid_layers: int=2, n_hidden: int=100, act: str='relu', \
                        dropout=0.0, requires_grad=True,layer_norm=False,batch_norm=False, bias=True):
        
        super().__init__()
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        assert not (layer_norm and batch_norm), 'Either layer_norm or batch_norm should be True'
        self.weight_mus  = nn.ParameterList([])
        self.bias_mus    = nn.ParameterList([])
        self.norms = nn.ModuleList([])
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.acts    = []
        self.act = act 
        self.bias = bias
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weight_mus.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.bias_mus.append(None if not bias else Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
            norm = nn.Identity()
            if i < n_hid_layers:
                if layer_norm:
                    norm = nn.LayerNorm(n_out)
                elif batch_norm:
                    norm = nn.BatchNorm1d(n_out)
            self.norms.append(norm)
        self.double()
        self.reset_parameters()

    @property
    def device(self):
        return self.weight_mus[0].device
    

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weight_mus,self.bias_mus)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            if self.bias:
                nn.init.uniform_(bias, -bound, bound)
        for norm in self.norms[:-1]:
            if isinstance(norm,nn.LayerNorm):
                norm.reset_parameters()


    def draw_f(self):
        """ 
            x=[N,n] & bnn=False ---> out=[N,n]
            x=[N,n] & L=1 ---> out=[N,n]
            x=[N,n] & L>1 ---> out=[L,N,n]
            x=[L,N,n] -------> out=[L,N,n]
        """
        def f(x):
            for (weight,bias,act,norm) in zip(self.weight_mus,self.bias_mus,self.acts,self.norms):
                x = act(norm(self.dropout(F.linear(x,weight.T,bias))))
            return x
        return f


    def forward(self, x):
        f = self.draw_f()
        return f(x)
