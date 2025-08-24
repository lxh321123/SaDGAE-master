from preprocess import preprocess
from GAE import GNN
from utils import *
from train import pretrain_gnn
from opt import args
from AE import AE
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.name == "goolam" or args.name == "Xin" or args.name == "Yan" or args.name == "test" or args.name == "Baron_Mouse":
    data_path = f"./dataset/{args.name}"
else:
    data_path = f"./dataset/{args.name}/data.h5"

args.save_ae_model_path = f"./model/{args.name}_ae_train.pkl"
args.save_gnn_model_path = f"./model/{args.name}_gnn_train.pkl"
x, real_label = preprocess(data_path, args.hvg)

n_cell, n_feature = x.shape
model = GNN(
    input_size=args.output_size,
    gnn_en_1=args.gnn_en_1,
    gnn_en_2=args.gnn_en_2,
    n_z=args.n_z,
    gnn_de_1=args.gnn_de_1,
    gnn_de_2=args.gnn_de_2,
).to(device)

ae = AE(
    input_size=n_feature,
    ae_en_1=args.ae_en_1,
    ae_en_2=args.ae_en_2,
    ae_en_3=args.ae_en_3,
    output_size=args.output_size,
    ae_de_1=args.ae_de_1,
    ae_de_2=args.ae_de_2,
    ae_de_3=args.ae_de_3,
    ).to(device)

pretrain_gnn(x, real_label, model, ae, device)