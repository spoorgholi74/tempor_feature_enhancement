#!/usr/bin/env python

"""Baseline for relative time embedding: learn regression model in terms of
relative time.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import torch
import torch.nn as nn
from torch.nn import functional as F



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.featureSize = 512

        ### concat
        self.fc1 = nn.Linear(self.featureSize,256)

        # self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.fc2 = nn.Linear(256, 128)
        self.fc_last = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = torch.sigmoid(self.fc_last(x))
        x = self.fc_last(x)
        return x.view(-1)
        #return x

    def embedded(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        #x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def create_model():
    torch.manual_seed(74)
    model = MLP()#.cuda()
    loss = nn.MSELoss(reduction='sum')#.cuda()
    #loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=0.0001)
    return model, loss, optimizer

