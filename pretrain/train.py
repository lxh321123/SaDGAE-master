import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from opt import args
from ZINB import ZINB
import torch.nn.functional as F
from utils import loss_cluster, eval, get_adj
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from tqdm import tqdm
color_green = "\033[92m"
color_red = "\033[91m"
color_yellow = "\033[93m"
color_pred = "\033[95m"
color_reset = "\033[0m"

def train(x, y, model, device):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    patience = 0
    best_ari = 0
    best_nmi = 0
    best_loss = torch.inf
    # for param in model.ae.parameters():
    #     param.requires_grad = False

    for epoch in tqdm(range(args.epochs), ncols=80, colour="red", desc="scDeGAEsa"):
        h_train = model.ae.ae_encoder(x_train)
        _, adj_train = get_adj(h_train)
        adj_train = torch.from_numpy(adj_train.astype(np.float32)).to(device)

        x_hat_train, pi_train, mu_train, theta_train, z_train, adj_hat_train, z_hat_train, h_train = model(x_train, adj_train, args.active)
        zinb = ZINB(x_train, pi_train, mu_train, theta_train)
        loss_zinb = zinb.loss()
        loss_mse1 = F.mse_loss(adj_hat_train, adj_train)
        loss_mse2 = F.mse_loss(z_hat_train, torch.spmm(adj_train, h_train))
        loss_mse3 = F.mse_loss(x_hat_train, x_train)

        loss_train = loss_zinb + loss_mse1 + loss_mse2 + loss_mse3
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        # lr_scheduler.step(loss)

        with torch.no_grad():
            h_test = model.ae.ae_encoder(x_test)
            _, adj_test = get_adj(h_test)
            adj_test = torch.from_numpy(adj_test.astype(np.float32)).to(device)

            x_hat_test, pi_test, mu_test, theta_test, z_test, adj_hat_test, z_hat_test, h_test = model(x_test,adj_test,args.active)
            zinb = ZINB(x_test, pi_test, mu_test, theta_test)
            loss_zinb = zinb.loss()
            loss_mse1 = F.mse_loss(adj_hat_test, adj_test)
            loss_mse2 = F.mse_loss(z_hat_test, torch.spmm(adj_test, h_test))
            loss_mse3 = F.mse_loss(x_hat_test, x_test)
            loss_test = loss_zinb + loss_mse1 + loss_mse2 + loss_mse3

        if (epoch + 1) % 10 == 0:
            y_pre = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10).fit_predict(z_test.cpu().detach().numpy())
            ari, nmi = eval(y_test, y_pre)
            # print(f"epoch is {epoch + 1}/{args.epochs}, loss train is {loss_train.item()}, loss test is {loss_test.item()}, ari is {ari}, nmi is {nmi}")
            # print(f"epoch is {color_green}{epoch + 1}/{args.epochs}{color_reset}, loss is {color_red}{loss.item()}{color_reset}, ari is {color_yellow}{ari}{color_reset}, nmi is {color_yellow}{nmi}{color_reset}")
            if ari > best_ari and nmi > best_nmi:
                best_ari = ari
                best_nmi = nmi

            if loss_test < best_loss:
                best_loss = loss_test
                patience = 0
                torch.save(model.state_dict(), args.save_pretrain_model_path)
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break

        # if epoch > 150:
        #    for param in model.ae.parameters():
        #        param.requires_grad = True

    print(f"best ari is {best_ari:.5f}, best nmi is {best_nmi:.5f}")
    # print(f"best ari is {color_pred}{best_ari:.5f}{color_reset}, best nmi is {color_pred}{best_nmi:.5f}{color_reset}")


