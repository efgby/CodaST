import os
import torch
import pandas as pd
import scanpy as sc
import random
import numpy as np
from sklearn import metrics
import multiprocessing as mp
from CodaST import network as CodaST
import matplotlib.pyplot as plt
import logging
import time
import threading
import sys
sc.settings.verbosity = 3 
logging.basicConfig(level=logging.INFO)

class TimeoutError(Exception):
    pass

def timeout_function(func, args=(), kwargs={}, timeout_duration=10):
    result = [None]
    error = [None]
    finished = [False]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
            finished[0] = True
        except Exception as e:
            error[0] = e
            finished[0] = True
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if not finished[0]:
        return None, TimeoutError(f"timeout (> {timeout_duration}s)")
    if error[0] is not None:
        return None, error[0]
    return result[0], None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
save_model_file = r'D:\gakkibot\CodaST\weights.pth'
os.environ['R_HOME'] = r'D:\R\R-4.2.3'
n_clusters = 7
dataset = '151673'
seed = 41
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

file_fold = 'D:\\gakkibot\\CodaST\\' + str(dataset) #please replace 'file_fold' with the download path
adata = sc.read_visium(file_fold, count_file=str(dataset)+'_filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

epochs = 600 
use_best_model = False  

model = CodaST.CodaST(adata, device=device, save_model_file=save_model_file, epochs=epochs,deconvolution=False)
adata = model.train()

from CodaST.utils import clustering
radius = 35
tool = 'mclust' # mclust, leiden, and louvain
if tool == 'mclust':
   clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
   clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

df_meta = pd.read_csv(file_fold + '/cluster_labels_'+str(dataset)+'.csv')
df_meta_layer = df_meta['ground_truth']
adata.obs['ground_truth'] = df_meta_layer.values
adata = adata[~pd.isnull(adata.obs['ground_truth'])]

ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
AMI = metrics.adjusted_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
FM = metrics.fowlkes_mallows_score(adata.obs['domain'], adata.obs['ground_truth'])
adata.uns['ARI'] = ARI
adata.uns['NMI'] = NMI
adata.uns['AMI'] = AMI
adata.uns['FM'] = FM

print('Dataset:', dataset)
print('ARI:', ARI)
print('NMI:', NMI)
print('AMI:', AMI)
print('FM:', FM)

result_dir = f'D:\\gakkibot\\CodaST\\results'
os.makedirs(result_dir, exist_ok=True)

try:
    plt.figure(figsize=(10, 8))
    sc.pl.spatial(adata,
                img_key="hires",
                color=["ground_truth", "domain"],
                title=["Ground truth", "ARI=%.4f"%ARI+" NMI=%.4f"%NMI+" AMI=%.4f"%AMI+" FM=%.4f"%FM],
                show=False)
    plt.savefig(f"{result_dir}/spatial_domain.png")
    plt.close()
except Exception as e:
    print(f"error: {e}")

try:
    sc.pp.neighbors(adata, use_rep='emb')
    sc.tl.umap(adata)
    print("UMAP finished")

    mask = ~pd.isnull(adata.obs['ground_truth'])
    nan_check = adata.obs['ground_truth'].astype(str).isin(['nan', 'NaN', 'NAN'])
    if nan_check.any():
        mask = mask & ~nan_check
    used_adata = adata[mask].copy()
    print(f"filtered cells: {used_adata.shape[0]}")
    
    plt.figure(figsize=(8, 6))
    sc.pl.umap(used_adata, color=['domain', 'ground_truth'], show=False)
    plt.savefig(f"{result_dir}/umap.png")
    plt.close()
    
    def compute_paga():
        sc.tl.paga(used_adata, groups='domain')
        return used_adata
    
    result, error = timeout_function(compute_paga, timeout_duration=300)
    
    if error is None:
        print("PAGA finished")
        plt.rcParams["figure.figsize"] = (4, 3)
        try:
            sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                       title=dataset+'_CodaST', legend_fontoutline=2, threshold=0.3, 
                       show=False)
            plt.savefig(f"{result_dir}/paga.png")
            plt.close()
        except Exception as e:
            print(f"PAGA error: {e}")
    else:
        if isinstance(error, TimeoutError):
            print("PAGA timeout,skip")
        else:
            print(f"PAGA error: {error}")

except Exception as e:
    print(f"UMAP or PAGA error: {e}")

print("finished!")