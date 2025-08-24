from opt import args
from preprocess import preprocess
from AE import AE
from train import pretrain_ae
from torch.utils.data import DataLoader, TensorDataset
from utils import *

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.name == "goolam" or args.name == "Xin" or args.name == "Yan" or args.name == "test" or args.name == "Baron_Mouse":
    dataset_path = f"./dataset/{args.name}"
else:
    dataset_path = f"./dataset/{args.name}/data.h5"

args.save_model_path = f"./model/{args.name}_ae_train.pkl"
x, y = preprocess(dataset_path, args.hvg)
# dataset = TensorDataset(x, real_label)
# data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

num_cell, num_feature = x.shape
model = AE(
    input_size=num_feature,
    ae_en_1=args.ae_en_1,
    ae_en_2=args.ae_en_2,
    ae_en_3=args.ae_en_3,
    output_size=args.output_size,
    ae_de_1=args.ae_de_1,
    ae_de_2=args.ae_de_2,
    ae_de_3=args.ae_de_3,
    ).to(device)

pretrain_ae(x, y, model, device)