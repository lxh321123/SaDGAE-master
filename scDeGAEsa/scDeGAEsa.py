import torch
import torch.nn as nn
import torch.nn.functional as F
from AE import AE
from utils import *
from opt import args
from GAE import GNN


class scDeGAEsa(nn.Module):
    def __init__(self, input_size, ae_en_1, ae_en_2, ae_en_3, output_size, ae_de_1, ae_de_2, ae_de_3, gnn_en_1, gnn_en_2, n_z, gnn_de_1, gnn_de_2):
        super(scDeGAEsa, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ae = AE(
            input_size=input_size,
            ae_en_1=ae_en_1,
            ae_en_2=ae_en_2,
            ae_en_3=ae_en_3,
            output_size=output_size,
            ae_de_1=ae_de_1,
            ae_de_2=ae_de_2,
            ae_de_3=ae_de_3).to(self.device)
        self.ae.load_state_dict(torch.load(args.save_ae_model_path))

        self.gnn = GNN(
            input_size=output_size,
            gnn_en_1=gnn_en_1,
            gnn_en_2=gnn_en_2,
            n_z=n_z,
            gnn_de_1=gnn_de_1,
            gnn_de_2=gnn_de_2,
        ).to(self.device)

        self.gnn.load_state_dict(torch.load(args.save_gnn_model_path))


    def forward(self, x, adj, active):
        x_hat, h, pi, mu, theta = self.ae(x)
        z, adj_hat, z_hat = self.gnn(h, adj, active)

        return x_hat, pi, mu, theta, z, adj_hat, z_hat, h