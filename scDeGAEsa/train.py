import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from opt import args
from ZINB import ZINB
import torch.nn.functional as F
from utils import loss_cluster, eval, get_adj
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from tqdm import tqdm
color_green = "\033[92m"
color_red = "\033[91m"
color_yellow = "\033[93m"
color_pred = "\033[95m"
color_reset = "\033[0m"

def train(x, y, model, device):
    x = torch.from_numpy(x).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    patience = 0
    best_ari = 0
    best_nmi = 0

    # for param in model.ae.parameters():
    #     param.requires_grad = False

    for epoch in tqdm(range(args.epochs), ncols=80, colour="blue", desc="scDeGAEsa"):
        h = model.ae.ae_encoder(x)
        if epoch == 0 or (epoch + 1) % 100 == 0:
            _, adj = get_adj(h)
            adj = torch.from_numpy(adj.astype(np.float32)).to(device)

        x_hat, pi, mu, theta, z, adj_hat, z_hat, h = model(x, adj, args.active)
        zinb = ZINB(x, pi, mu, theta)
        loss_zinb = zinb.loss()
        loss_mse1 = F.mse_loss(adj_hat, adj)
        loss_mse2 = F.mse_loss(z_hat, torch.spmm(adj, h))
        loss_mse3 = F.mse_loss(x_hat, x)

        if epoch == 0:
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
            kmeans.fit(z.cpu().detach().numpy())
            centers = kmeans.cluster_centers_
            centers = torch.from_numpy(centers).float().to(device)
        else:
            labels = torch.argmax(q, dim=1)
            centers_hat = torch.zeros_like(centers).to(device)
            for l in range(q.size(1)):
                z_l = z[l == labels]
                q_l = q[l == labels, l].unsqueeze(1)
                if q_l.sum() != 0:
                    centers_hat[l] = (q_l * z_l).sum(dim=0) / q_l.sum()
            centers = centers_hat.detach().clone()

        loss_clu, q = loss_cluster(z, centers)
        loss = 0.1 * loss_zinb + 0.3 * loss_mse1 + 0.3 * loss_mse2 + 0.3 * loss_mse3 + 3 * loss_clu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step(loss)

        if (epoch + 1) % 10 == 0:
            if epoch == 0:
                labels = torch.argmax(q, dim=1)
            ari, nmi = eval(y, labels.cpu().detach().numpy())
            # print(f"epoch is {epoch + 1}/{args.epochs}, loss is {loss.item()}, ari is {ari}, nmi is {nmi}")
            # print(f"epoch is {color_green}{epoch + 1}/{args.epochs}{color_reset}, loss is {color_red}{loss.item()}{color_reset}, ari is {color_yellow}{ari}{color_reset}, nmi is {color_yellow}{nmi}{color_reset}")
            if ari > best_ari and nmi > best_nmi:
                best_ari = ari
                best_nmi = nmi
                torch.save(model.state_dict(), args.save_final_model_path)
        # if epoch > 150:
        #    for param in model.ae.parameters():
        #        param.requires_grad = True

    print(f"best ari is {best_ari:.5f}, best nmi is {best_nmi:.5f}")
    # print(f"best ari is {color_pred}{best_ari:.5f}{color_reset}, best nmi is {color_pred}{best_nmi:.5f}{color_reset}")


