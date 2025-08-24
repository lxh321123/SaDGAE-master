import torch
from tqdm import tqdm
from opt import args
from ZINB import ZINB
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
color_green = "\033[92m"
color_red = "\033[91m"
color_yellow = "\033[93m"
color_reset = "\033[0m"

def pretrain_ae(x, y, model, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    patience = 0
    best_test_loss = np.inf
    best_ari = 0
    best_nmi = 0

    for epoch in range(args.epoch):
        x_train_hat, h_train, pi_train, mu_train, theta_train = model(x_train)
        zinb_train = ZINB(x_train, pi_train, mu_train, theta_train)
        loss_mse_train = F.mse_loss(x_train_hat, x_train)
        loss_train = zinb_train.loss() + loss_mse_train

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            x_test_hat, h_test, pi_test, mu_test, theta_test = model(x_test)
            zinb_test = ZINB(x_test, pi_test, mu_test, theta_test)
            loss_mse_test = F.mse_loss(x_test_hat, x_test)
            loss_test = zinb_test.loss() + loss_mse_test

            if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epoch:
                y_pred = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10).fit_predict(h_test.cpu().detach().numpy())
                ari, nmi = eval(y_test, y_pred)
                # print(f"epoch is {epoch + 1}/{args.epoch}, train loss is {loss_train.item()}, test loss is {loss_test.item()}, ari is {ari}, nmi is {nmi}")
                print(f"epoch is {color_green}{epoch + 1}/{args.epoch}{color_reset}, train loss is {color_red}{loss_train.item()}{color_reset}, "
                      f"test loss is {color_red}{loss_test.item()}{color_reset}, ari is {color_yellow}{ari}{color_reset}, nmi is {color_yellow}{nmi}{color_reset}")

                if ari > best_ari and nmi > best_nmi:
                    best_ari = ari
                    best_nmi = nmi

                if loss_test < best_test_loss:
                    best_test_loss = loss_test
                    patience = 0
                    torch.save(model.state_dict(), args.save_model_path)
                else:
                    patience += 1
                    if patience >= args.patience:
                        # print("Early stopping")
                        print(f"{color_yellow}Early stopping{color_reset}")
                        break

    print(f"best ari:{best_ari}, best nmi:{best_nmi}")
    # print(f"best ari:{color_yellow}{best_ari}{color_reset}, best nmi:{color_yellow}{best_nmi}{color_reset}")