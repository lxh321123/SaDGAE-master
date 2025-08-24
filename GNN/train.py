import os
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from sklearn.cluster import KMeans
from opt import args
color_green = "\033[92m"
color_red = "\033[91m"
color_yellow = "\033[93m"
color_pred = "\033[95m"
color_reset = "\033[0m"



def pretrain_gnn(x, y, model, ae, device):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    patience = 0
    best_test_loss = np.inf
    best_ari = 0
    best_nmi = 0

    _, x_train, _, _, _ = ae(x_train)
    _, x_test, _, _, _ = ae(x_test)

    _, adj_train= get_adj(x_train)
    adj_train = torch.from_numpy(adj_train.astype(np.float32)).to(device)

    _, adj_test = get_adj(x_test)
    adj_test = torch.from_numpy(adj_test.astype(np.float32)).to(device)

    for epoch in range(args.epochs):
        z_train, adj_hat_train, x_hat_train = model(x_train, adj_train, args.active)
        loss_train1 = F.mse_loss(adj_hat_train, adj_train)
        loss_train2 = F.mse_loss(x_hat_train, torch.spmm(adj_train, x_train))
        loss_train = loss_train1 + loss_train2

        optimizer.zero_grad()
        loss_train.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            z_test, adj_hat_test, x_hat_test = model(x_test, adj_test, args.active)
            loss_test1 = F.mse_loss(adj_hat_test, adj_test)
            loss_test2 = F.mse_loss(x_hat_test, torch.spmm(adj_test, x_test))

            loss_test = loss_test1 + loss_test2

            if (epoch + 1) % 50 == 0:
                y_pred = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10).fit_predict(z_test.cpu().detach().numpy())
                ari, nmi = eval(y_test, y_pred)
                # print(f"epoch is {epoch + 1}/{args.epochs}, train loss is {loss_train.item()}, test loss is {loss_test.item()}, ari is {ari}, nmi is {nmi}")
                print(
                    f"epoch is {color_green}{epoch + 1}/{args.epochs}{color_reset}, train loss is {color_red}{loss_train.item()}{color_reset}, "
                    f"test loss is {color_red}{loss_test.item()}{color_reset}, ari is {color_yellow}{ari}{color_reset}, nmi is {color_yellow}{nmi}{color_reset}"
                )
                if ari > best_ari and nmi > best_nmi:
                    best_ari = ari
                    best_nmi = nmi
                if loss_test < best_test_loss:
                    best_test_loss = loss_test
                    patience = 0
                    torch.save(model.state_dict(), args.save_gnn_model_path)
                else:
                    patience += 1
                    if patience >= args.patience:
                        # print(f"Early stopping")
                        print(f"{color_yellow}Early stopping{color_reset}")
                        break

        # if (epoch + 1) >= 500:
        #     for param in model.ae.parameters():
        #         param.requires_grad = True

    # print(f"best ari:{best_ari}, best nmi:{best_nmi}")
    print(f"best ari:{color_yellow}{best_ari}{color_reset}, best nmi:{color_yellow}{best_nmi}{color_reset}")