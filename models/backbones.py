import torch
import torch.nn as nn
from functools import partial


class BackBone(nn.Module):
    def __init__(self, args):
        super(BackBone, self).__init__()
        if args.model_type == 'small':
            from .vit import vit_small
            self.encoder = vit_small()
            
        else:
            raise ValueError('Unknown model type')

    def forward(self, support, query):
        # feature extraction
        support = self.encoder(support)
        query = self.encoder(query)
        return support, query

