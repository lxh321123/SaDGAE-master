import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class GNN_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GNN_layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.act = nn.LeakyReLU()
        self.weight = nn.Parameter(torch.FloatTensor(output_size, input_size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        if active:
            support = self.act(F.linear(features, self.weight))
        else:
            support = F.linear(features, self.weight)
        output = torch.spmm(adj, support)
        return output

class GNN_encoder(nn.Module):
    def __init__(self, input_size, gnn_en_1, gnn_en_2, n_z):
        super(GNN_encoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_layer1 = GNN_layer(input_size, gnn_en_1).to(self.device)
        self.gnn_layer2 = GNN_layer(gnn_en_1, gnn_en_2).to(self.device)
        self.gnn_layer3 = GNN_layer(gnn_en_2, n_z).to(self.device)
        self.s = nn.Sigmoid()

    def forward(self, x, adj, active):
        z = self.gnn_layer1(x, adj, active)
        z = self.gnn_layer2(z, adj, active)
        z = self.gnn_layer3(z, adj, active=False)
        adj_hat_en = self.s(torch.mm(z, z.t()))
        return z, adj_hat_en

class GNN_decoder(nn.Module):
    def __init__(self, n_z, gnn_de_1, gnn_de_2, input_size):
        super(GNN_decoder, self).__init__()
        self.gnn_layer1 = GNN_layer(n_z, gnn_de_1)
        self.gnn_layer2 = GNN_layer(gnn_de_1, gnn_de_2)
        self.gnn_layer3 = GNN_layer(gnn_de_2, input_size)
        self.s = nn.Sigmoid()

    def forward(self, z, adj, active):
        x_hat = self.gnn_layer1(z, adj, active)
        x_hat = self.gnn_layer2(x_hat, adj, active)
        x_hat = self.gnn_layer3(x_hat, adj, active)
        adj_hat_de = self.s(torch.mm(x_hat, x_hat.t()))
        return x_hat, adj_hat_de

class GNN(nn.Module):
    def __init__(self, input_size, gnn_en_1, gnn_en_2, n_z, gnn_de_1, gnn_de_2):
        super(GNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn_encoder = GNN_encoder(input_size, gnn_en_1, gnn_en_2, n_z).to(self.device)
        self.gnn_decoder = GNN_decoder(n_z, gnn_de_1, gnn_de_2, input_size).to(self.device)

    def forward(self, x, adj, active):
        z, adj_hat_en = self.gnn_encoder(x, adj, active)
        x_hat, adj_hat_de = self.gnn_decoder(z, adj, active)
        adj_hat = adj_hat_en + adj_hat_de

        return F.log_softmax(z, dim=-1), adj_hat, x_hat