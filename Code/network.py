import torch
import os
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, \
    construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed
import time
import random
import numpy as np
from .model import Encoder, Encoder_sparse,Encoder_sc,Encoder_map
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from .layers import Clusterator, Discriminator_cluster  
from .utils import Transfer_pytorch_Data
import scipy.sparse as sp
from torch.autograd import Variable
from sklearn.cluster import KMeans

class CodaST():
    def __init__(self,
                 adata,
                 adata_sc=None,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 learning_rate_sc=0.01,
                 weight_decay=0.00,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 random_seed=41,
                 alpha=10,
                 datatype='10X',
                 save_model_file=None,
                 lambda_nccl=1.2, 
                 lambda_intra=0.01,  
                 deconvolution=False,
                 sigma=0.2,  # 添加 sigma 参数，用于 IFL-GCL 语义权重
                 ):
        self.adata = adata.copy() if adata is not None else None
        self.adata_sc = adata_sc.copy() if adata_sc is not None else None
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.random_seed = random_seed
        self.alpha = alpha
        self.datatype = datatype
        self.tau = 0.5
        self.save_model_file = save_model_file
        
        if self.save_model_file is None:
            self.save_model_file = os.path.join(os.getcwd(), "CodaST_models", "weights.pth")
            os.makedirs(os.path.dirname(self.save_model_file), exist_ok=True)
        
        self.deconvolution = deconvolution
        self.learning_rate_sc = learning_rate_sc
        self.lambda_nccl = lambda_nccl
        self.lambda_intra = lambda_intra 
        self.sigma = sigma  # 存储 sigma 参数
        self.n_clusters = 7
        
        fix_seed(self.random_seed)
        self.fc1 = torch.nn.Linear(self.dim_output, 128).to(self.device)
        self.fc2 = torch.nn.Linear(128, self.dim_output).to(self.device)

        if 'highly_variable' not in self.adata.var.columns:
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                construct_interaction_KNN(self.adata)
            else:
                construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            add_contrastive_label(self.adata)
            
        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy())
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy())
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL'])
        self.adj = self.adata.obsm['adj']
        
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.features = self.features.to(self.device)
        self.features_a = self.features_a.to(self.device)
        self.label_CSL = self.label_CSL.to(self.device)
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)

        self.dim_output = dim_output

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            self.adj = preprocess_adj_sparse(self.adj).to(self.device)
        else:
            # standard version
            fix_seed(self.random_seed)
            self.adj = preprocess_adj(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)

        if 'Spatial_Net' not in self.adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        self.data = Transfer_pytorch_Data(self.adata).to(self.device)

        if self.deconvolution:
            self.adata_sc = self.adata_sc.copy()
            if  isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
                self.feat_sp = adata.X.toarray()[:, ]
            else:
                self.feat_sp = adata.X[:, ]
            if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
                self.feat_sc = adata_sc.X.toarray()[:, ]
            else:
                self.feat_sc = adata_sc.X[:, ]             
            self.feat_sc = pd.DataFrame(self.feat_sc).fillna(0).values
            self.feat_sp = pd.DataFrame(self.feat_sp).fillna(0).values
            self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
            self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
            
            if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 
            self.n_cell = adata_sc.n_obs
            self.n_spot = adata.n_obs


    def compute_neighbor_mask(self, adj_matrix):
        mask = (adj_matrix > 0).float()
        return mask

    def neighbor_consistency_loss(self, embeddings, neighbor_mask, temperature=0.3, sigma=0.5):
       
        z = F.normalize(embeddings, dim=1)          # [N, D]
        sim = torch.matmul(z, z.T) / temperature    # [N, N] 内积相似度 / τ
        with torch.no_grad():
            sim_cos = torch.matmul(z, z.T)          # [N, N]
            w_ij = torch.exp(sim_cos / sigma)

        exp_sim = torch.exp(sim) * (1 - torch.eye(z.shape[0], device=z.device))  # [N, N]
        weighted_pos = w_ij * exp_sim * neighbor_mask
        numerator = weighted_pos.sum(dim=1, keepdim=True)  # [N,1]
        denominator = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        ratio = numerator / denominator.clamp(min=1e-8)
        valid = neighbor_mask.sum(1) > 0
        loss_i = -torch.log(ratio[valid] + 1e-8)
        loss = loss_i.mean()
        return loss

    
    def compute_intra_inter_loss(self, embeddings, soft_assign):
        N, D = embeddings.shape
        K = soft_assign.shape[1]
        
        assign_sum = soft_assign.sum(0)  # [K]
        normalized_assign = soft_assign / (assign_sum.unsqueeze(0) + 1e-8)  # [N, K]
        
        centers = torch.matmul(embeddings.t(), normalized_assign).t()  # [K, D]

        z_center = torch.matmul(soft_assign, centers)  # [N, D]
        
        loss_intra = F.mse_loss(embeddings, z_center)
        centers = F.normalize(centers, dim=1)  
        dist_mat = torch.cdist(centers, centers, p=2)  # [K, K]
        
        mask = 1.0 - torch.eye(K, device=centers.device)        
        return loss_intra, centers
    
    def generate_soft_assignment(self, embeddings, temperature=0.1):
        with torch.no_grad():
            X = embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed).fit(X)
            centroids = torch.tensor(kmeans.cluster_centers_, 
                                    device=embeddings.device, 
                                    dtype=embeddings.dtype)
        
        dist = torch.cdist(embeddings, centroids, p=2)  # [N, K]
        
        sim = -dist / temperature
        soft_assign = F.softmax(sim, dim=1)  # [N, K]
        
        return soft_assign
        
    def train(self):
        fix_seed(self.random_seed)
        actual_dim = self.features.shape[1]
        self.dim_input = actual_dim        
        if self.datatype in ['Stereo', 'Slide']:
            self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        else:
            self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
    
        self.cluster_layer = nn.Linear(self.dim_output, self.n_clusters).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.cluster_layer.parameters()),
            self.learning_rate,
            weight_decay=self.weight_decay
        )
                
        fix_seed(self.random_seed)
      
        print('Begin to train ST data...')
        self.model.train()
        
        best_loss = float('inf')
        best_state = None
        self.soft_assign = None
        self.cluster_centers = None
        
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            fix_seed(self.random_seed)
            self.features_a = permutation(self.features)
            
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                self.hiden_feat, self.emb = self.model(self.features, self.features_a,
                                                       self.data.edge_index)[0:2]
            except RuntimeError as e:
                if "shapes cannot be multiplied" in str(e):
                    self.dim_input = self.features.shape[1]
                    if self.datatype in ['Stereo', 'Slide']:
                        self.model = Encoder_sparse(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
                    else:
                        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
                    
                    self.optimizer = torch.optim.Adam(
                        list(self.model.parameters()) + list(self.cluster_layer.parameters()),
                        self.learning_rate,
                        weight_decay=self.weight_decay
                    )
                    
                    self.hiden_feat, self.emb = self.model(self.features, self.features_a,
                                                           self.data.edge_index)[0:2]
                else:
                    raise
            
            # if epoch % self.cluster_update_interval == 0:
            #     self.soft_assign = self.generate_soft_assignment(
            #         self.hiden_feat.detach(),
            #         temperature=0.1
            #     )
                
                logits = self.cluster_layer(self.hiden_feat.detach())
                self.soft_assign = F.softmax(logits, dim=1)
        
            
            if self.soft_assign is not None:
                self.loss_intra, self.cluster_centers = self.compute_intra_inter_loss(
                    self.hiden_feat, self.soft_assign
                )
            else:
                self.loss_intra = torch.tensor(0.0).to(self.device)

            self.loss_feat = F.mse_loss(self.features, self.emb)
           
            neighbor_mask = self.compute_neighbor_mask(self.graph_neigh)
            self.loss_nccl = self.neighbor_consistency_loss(self.hiden_feat, neighbor_mask, sigma=self.sigma)  # 传入 sigma 参数

            loss = (self.alpha * self.loss_feat  
                    + self.lambda_nccl * self.loss_nccl      
                    + self.lambda_intra * self.loss_intra     
                    )
      
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    'encoder': {k: v.cpu() for k, v in self.model.state_dict().items()},
                    'cluster': {k: v.cpu() for k, v in self.cluster_layer.state_dict().items()}
                }

        print("Optimization finished for ST data!")
        
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state['encoder'].items()})
            self.cluster_layer.load_state_dict({k: v.to(self.device) for k, v in best_state['cluster'].items()})
            
        self.save_model(save_model_file=self.save_model_file)

        with torch.no_grad():
            self.model.eval()
            if self.deconvolution:
                self.emb = self.model(self.features, self.features_a, self.data.edge_index)[1]
                return self.emb  
            else:
                if self.datatype in ['Stereo', 'Slide']:
                    self.emb_rec = self.model(self.features, self.features_a, self.data.edge_index)[1]
                    self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                else:
                    fix_seed(self.random_seed)
                    self.emb_rec = self.model(self.features, self.features_a, self.data.edge_index)[
                        1].detach().cpu().numpy()
            
                final_feat = self.model(self.features, self.features_a, self.data.edge_index)[0]
                final_logits = self.cluster_layer(final_feat) 
                final_probs = F.softmax(final_logits, dim=1)
                cluster_assignments = torch.argmax(final_probs, dim=1).cpu().numpy()
            
                self.adata.obsm['emb'] = self.emb_rec
                self.adata.obs['nccl_cluster'] = cluster_assignments

                return self.adata
   
    def train_sc(self):
        actual_dim_sc = self.feat_sc.shape[1]
        self.dim_input = actual_dim_sc
    
        self.model_sc = Encoder_sc(self.dim_input, self.dim_output).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        
        print('Begin to train scRNA data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            
            emb_sc,x_rec = self.model_sc(self.feat_sc)
            loss = F.mse_loss(x_rec, self.feat_sc)
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
            
        print("Optimization finished for cell representation learning!")
        
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc= self.model_sc(self.feat_sc)[1]
            return emb_sc
        
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        print(emb_sc.shape,emb_sp.shape)
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        emb_sp = F.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = F.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        
        self.model_map = Encoder_map(self.n_cell, self.n_spot).to(self.device)  
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        print('Begin to learn mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()
            map_probs = F.softmax(self.map_matrix, dim=1)
            pred_sp = torch.matmul(map_probs.t(), emb_sc)
            loss_recon = F.mse_loss(pred_sp, emb_sp, reduction='mean')
            loss_NCE = self.Noise_Cross_Entropy(pred_sp, emb_sp)
            lamda1 = getattr(self, "lamda1", 1.0)
            lamda2 = getattr(self, "lamda2", 0.1)
            loss = lamda1 * loss_recon + lamda2 * loss_NCE  

            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step()
        
        print("Mapping matrix learning finished!")
        
        # take final softmax w/o computing gradients
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = F.softmax(self.map_matrix, dim=1).cpu().numpy() # dim=1: normalization by cell
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T # spot x cell
            return self.adata, self.adata_sc

    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
 
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):  #pres_sp: spot x gene; emb_sp: spot x gene
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

        return M        
    def extract_top_value(self, map_matrix, retain_percent=0.1):
        top_k = int(retain_percent * map_matrix.shape[1])
        output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
        
        return output

    def project_cell_to_spot(self, retain_percent=0.1):
        adata = self.adata
        adata_sc = self.adata_sc
        if 'map_matrix' not in adata.obsm:
            raise KeyError("空间数据 adata.obsm 不包含 'map_matrix'，请先运行 train_map。")
        if adata_sc is None:
            raise ValueError("未提供单细胞数据 adata_sc。")
        # read map matrix 
        map_matrix = adata.obsm['map_matrix']   # spot x cell

        # extract top-k values for each spot
        top_k  = int(retain_percent * map_matrix.shape[1])
        map_matrix = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)    
    
        if 'celltype' in adata_sc.obs.columns:
            label = 'celltype'
        elif 'cell_type' in adata_sc.obs.columns:
            label = 'cell_type'
        else:
            raise KeyError(
                f"单细胞数据 adata_sc.obs 不包含 'celltype' 或 'cell_type' 列。"
                f"\n当前可用的 obs 列为: {list(adata_sc.obs.columns)}"
            )
    
        # construct cell type matrix
        matrix_celltype = self.construct_celltype_matrix(adata_sc, label=label)
        matrix_celltype = matrix_celltype.values

        # projection by spot-level
        matrix_projection = map_matrix.dot(matrix_celltype)

        # rename cell types
        celltype = list(adata_sc.obs[label].unique())
        celltype = [str(s) for s in celltype]
        celltype.sort()
        #celltype = [s.replace(' ', '_') for s in celltype]
        df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=celltype)  # spot x cell type

        #normalize by row (spot)
        df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

        #add projection results to adata
        adata.obs[df_projection.columns] = df_projection
        return adata

    def construct_celltype_matrix(self, adata_sc, label='celltype'):
        if label not in adata_sc.obs.columns:
            raise KeyError(
                f"单细胞数据 adata_sc.obs 不包含 '{label}' 列。"
                f"\n当前可用的 obs 列为: {list(adata_sc.obs.columns)}"
            )
        n_type = len(list(adata_sc.obs[label].unique()))
        zeros = np.zeros([adata_sc.n_obs, n_type])
        celltype = list(adata_sc.obs[label].unique())
        celltype = [str(s) for s in celltype]
        celltype.sort()
        mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=celltype)
        for cell in list(adata_sc.obs_names):
            ctype = adata_sc.obs.loc[cell, label]
            mat.loc[cell, str(ctype)] = 1
        return mat

    ###—————————————————————————————————————[3]function for save and load model weights————————————————————————————————————————————————————————————###
    def save_model(
        self,
        save_model_file
        ):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(
        self,
        save_model_file
        ):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

