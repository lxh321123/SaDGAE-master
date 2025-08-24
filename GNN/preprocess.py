import os.path
import h5py
import numpy as np
import pandas as pd
import scipy as sp
import torch
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import utils as utils
from opt import args

color_green = "\033[92m"
color_reset = "\033[0m"
# 确保数据是NumPy数组，并且如果数据类型是字节类型，则将其解码为字符串。
def read_clean(data):
    """
    :param data: 需要清理的数据
    :return: 清理后的数据
    """
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    """
    :param group: HDF5文件的组
    :return: 返回从组中读取的数据字典
    """
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def read_data(filename, sparsify=False, skip_exprs=False):
    """
    :param filename: 数据文件的路径
    :param sparsify: 是否将表达矩阵转换为稀疏矩阵
    :param skip_exprs: 是否跳过读取表达矩阵
    :return: 表达矩阵、细胞元数据、基因元数据和未归一化数据
    """

    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

def preprocess(name, hvg):
    if args.name == "Xin" or args.name == "Yan" or args.name == "Baron_Mouse":
        mat = pd.read_csv(os.path.join(name, "count.csv"))
        mat = mat.to_numpy()[:, 1:]
        real_label = pd.read_csv(os.path.join(name, "label.csv"))["cell_type1"].values
        real_label = LabelEncoder().fit_transform(real_label)
    elif args.name == "goolam":
        mat = pd.read_csv(os.path.join(name, "count.csv"), header=None)
        real_label = pd.read_csv(os.path.join(name, "label.csv"), header=None).iloc[:, 0]
        mat = np.array(mat).T
    elif (args.name == "Human1" or args.name == "Human2" or args.name == "Human3" or args.name == "Human4" or args.name == "Zeisel" or args.name == "Mouse1"
          or args.name == "Mouse2" or args.name == "Human_kidney" or args.name == "HumanLiver" or args.name == "10X_PBMC"):
        with h5py.File(name, "r") as f:
            mat = f["X"][:]
            real_label = f["Y"][:]
            mat = np.array(mat)
    else:
        mat, obs, var, uns = read_data(name, sparsify=False, skip_exprs=False)
        mat = np.array(mat.todense())
        print(f"the size of {args.name} is {mat.shape}")
        real_label = obs['cell_type1'].values
        real_label = LabelEncoder().fit_transform(real_label)

    adata = AnnData(mat, dtype=np.float32)

    cell_num = adata.shape[0]

    sc.pp.filter_cells(adata, min_genes=3)
    sc.pp.filter_genes(adata, min_cells=int(cell_num*0.2))

    # # 计算中位数
    # total_sum = adata.X.sum(axis=1)
    # median = np.median(total_sum)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
    adata = adata[:, adata.var["highly_variable"]].copy()
    x = np.array(adata.X)
    print(x.shape)
    return x, real_label
