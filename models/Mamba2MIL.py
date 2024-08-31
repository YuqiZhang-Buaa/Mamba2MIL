"""
Mamba2MIL
"""
import numpy as np
import torch
import torch.nn as nn
import math
from math import ceil
from mamba_new.mamba_ssm import Mamba2
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn, einsum

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class TransposeTokenReEmbedding:
    @staticmethod
    def transpose_normal_padding(x, dim):
        B, N, C = x.shape
        x_ = rearrange(x, "b (k w) d -> b (w k) d", w = dim)
        return x_

class Mamba2MIL(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2):
        super(SpeMIL, self).__init__()
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.layers_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        self.survival = survival

        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    # nn.LayerNorm(512),
                    Mamba2(
                        d_model=512,
                        d_state=64,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )
        for _ in range(layer):
            self.layers_1.append(
                nn.Sequential(
                    # nn.LayerNorm(512),
                    Mamba2(
                        d_model=512,
                        d_state=64,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )
        for _ in range(layer):
            self.layers_2.append(
                nn.Sequential(
                    # nn.LayerNorm(512),
                    Mamba2(
                        d_model=512,
                        d_state=64,  
                        d_conv=4,    
                        expand=2,
                    ),
                    )
            )

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 256]

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H

        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 256]
        
        h_o = h

        for layer in self.layers:
            h_ = h
            h = layer[0](h)
            # h = layer[1](h)
            h = h + h_
        h_0 = h

        h = h_o.flip([-1])
        for layer in self.layers_1:
            h_ = h
            h = layer[0](h)
            # h = layer[1](h)
            h = h + h_
        h_1 = h

        h = TransposeTokenReEmbedding.transpose_normal_padding(h_o, _W)
        for layer in self.layers_2:
            h_ = h
            h = layer[0](h)
            # h = layer[1](h)
            h = h + h_
        h_2 = h

        h = torch.cat((h_0, h_1, h_2), dim=1) 
        # print('h:', h)
        # print('-'*10)
        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)
        # print('h:', h)

        logits = self.classifier(h)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = None
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        self.layers_1  = self.layers_1.to(device)
        self.layers_2  = self.layers_2.to(device)
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)