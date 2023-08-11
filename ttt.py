
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
device = torch.device("cuda")
g.ndata['feat'] = torch.arange(2708)
print(torch.__version__)
g = g.to(device)