import random

import numpy as np
import scanpy as sc
import scipy.sparse
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

"""主要目的是提供一系列辅助函数和类，用于处理和分析单细胞RNA测序数据。"""
"""这个文件可能包含各种数据处理和辅助函数，特别是涉及基因表达矩阵的处理，
比如归一化、基因筛选等。你需要找到负责高变基因筛选的函数，并确保该部分代码可以顺利应用到你的新模型中。"""


class dotdict(dict):
    """让后续代码操作字典时更直观"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Compute cluster centroids, which is the mean of all points in one cluster.
# data表示基因表达数据矩阵 细胞×基因
# labels聚类标签，每个细胞（行）有一个对应的聚类标签
def computeCentroids(data, labels):
    """计算聚类中心，即一个聚类中所有点的平均值。"""
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])


# args.dataname 数据集名称,决定使用的聚类方法，若数据集名称在给定列表中，则使用SpectralClustering(基于预计算的邻接矩阵adj_n)，否则使用KMeans
# cluster_number 聚类的簇数量
# 质心通过 computeCentroids 或 cluster_model.cluster_centers_ 返回
def init_center(args, Y, adj_n, cluster_number):
    # Cluster center initialization
    """聚类中心初始化 根据传入的参数，使用KMeans或SpectralClustering初始化聚类中心"""
    from sklearn.cluster import KMeans, SpectralClustering
    if args.dataname in ["camp2", "Quake_Smart-seq2_Lung", "Muraro", "Adam", "Quake_10x_Limb_Muscle",
                         "Quake_Smart-seq2_Heart",
                         "Young", "Plasschaert", "Quake_10x_Spleen", "Chen", "Tosches turtle"]:
        labels = SpectralClustering(n_clusters=cluster_number, affinity="precomputed", assign_labels="discretize",
                                    n_init=20).fit_predict(adj_n)
        centers = computeCentroids(Y, labels)
    else:
        cluster_model = KMeans(n_clusters=cluster_number, n_init=20)
        labels = cluster_model.fit_predict(Y)
        centers = cluster_model.cluster_centers_
    return centers


# adata_ 包含单细胞基因表达矩阵的AnnData对象
# n_clusters 期望的簇数量
# 函数通过二分法不断调整Leiden 算法的分辨率，直到找到与期望簇数最接近的结果
def find_resolution(adata_, n_clusters, random=0):
    """根据期望的簇数量动态调整 Leiden 算法的分辨率参数。"""
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]

    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions) / 2
        sc.tl.leiden(adata, resolution=current_res, random_state=random)
        labels = adata.obs['leiden']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res

        iteration = iteration + 1

    return current_res


def densify(arr):
    """将稀疏矩阵转换为密集矩阵"""
    # 如果输入是scipy.sparse稀疏矩阵,则转化为密集矩阵,否则直接返回原数组
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    """一个装饰器函数，用于确保传入的数组不为空时再执行操作"""

    # 当数组大小不为0时，执行传入的函数fn
    def _fn(x):
        if x.size:
            return fn(x)
        # 否则直接返回转换后的数据类型
        return x.astype(dtype)

    return _fn


# X 基因表达矩阵
# dim PCA的目标维度，默认为10。函数返回降维后的矩阵
def dopca(X, dim=10):
    """对数据X进行主成分分析PCA，并降维到dim维度"""
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


# 将字节编码的字符串转换为普通的 UTF-8 字符串
# 使用empty_safe确保数组不为空，再使用np.vectorize 对数组中的每个元素进行解码
decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def eval(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi

def plot(loss_list, plt, label, style="ro-"):
    plt.plot(range(len(loss_list)), loss_list, style, label=label)