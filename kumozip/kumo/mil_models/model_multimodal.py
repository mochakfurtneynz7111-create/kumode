import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import ot
import numpy as np

from .components import SNN_Block, MMAttentionLayer, FeedForward, FeedForwardEnsemble, process_surv, Attn_Net_Gated,MultiheadAttention,BilinearFusion
from .components import *


def init_per_path_model(omic_sizes, hidden_dim=256):
    """
    Create a list of SNNs, one for each pathway

    Args:
        omic_sizes: List of integers, each indicating number of genes per prototype
    """
    hidden = [hidden_dim, hidden_dim]
    sig_networks = []
    for input_dim in omic_sizes:
        fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        sig_networks.append(nn.Sequential(*fc_omic))
    sig_networks = nn.ModuleList(sig_networks)

    return sig_networks

def agg_histo(X, agg_mode='mean'):
    """
    Aggregating histology
    """
    if agg_mode == 'mean':
        out = torch.mean(X, dim=1)
    elif agg_mode == 'cat':
        out = X.reshape(X.shape[0], -1)
    else:
        raise NotImplementedError(f"Not implemented for {agg_mode}")

    return out


class OT_Attn(nn.Module):
    """
    Optimal transport
    """
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)

    def normalize_feature(self, x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (B, N, D)
            weight2 : (B, M, D)

        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-uot-l2":
            a, b = torch.tensor(ot.unif(weight1.size()[0])).cuda(), torch.tensor(ot.unif(weight2.size()[0])).cuda()
            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)

            cost_map_detach = self.cost_map
            M_cost = cost_map_detach / cost_map_detach.max()

            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b,
                                                           M=M_cost.double(), reg=self.ot_reg,
                                                           reg_m=self.ot_tau)

            dist = self.cost_map * flow  # (N, M)
            # dist = torch.sum(dist)  # (1,) float
            return flow
        else:
            raise NotImplementedError(f"Not implemented for {self.impl}!")

    def forward(self, x, y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''

        pi = [self.OT(self.normalize_feature(x[idx]), self.normalize_feature(y[idx])) for idx in range(len(x))]
        pi = torch.stack(pi, dim=0)

        return pi.transpose(2, 1)


def construct_proto_embedding(path_proj_dim, append_embed='modality', numOfproto_histo=16, numOfproto_omics=50):
    """
    Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings 
    """
    if append_embed == 'modality':  # One-hot encoding for two modalities
        path_proj_dim_new = path_proj_dim + 2

        histo_embedding = torch.tensor([[[1, 0]]]).repeat(1, numOfproto_histo, 1)  # (1, numOfproto, 2)
        gene_embedding = torch.tensor([[[0, 1]]]).repeat(1, numOfproto_omics, 1)  # (1, len(omic_sizes),2 )

    elif append_embed == 'proto':
        path_proj_dim_new = path_proj_dim + numOfproto_histo + numOfproto_omics
        embedding = torch.eye(numOfproto_histo + numOfproto_omics).unsqueeze(0)

        histo_embedding = embedding[:, :numOfproto_histo, :]  # (1, numOfproto, numOftotalproto)
        gene_embedding = embedding[:, numOfproto_histo:, :]  # (1, len(omic_sizes), numOftotalproto)

    elif append_embed == 'random':
        append_dim = 32
        path_proj_dim_new = path_proj_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_histo, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, numOfproto_omics, append_dim), requires_grad=True)

    else:
        path_proj_dim_new = path_proj_dim
        histo_embedding = None
        gene_embedding = None

    return path_proj_dim_new, histo_embedding, gene_embedding


class coattn(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=1024,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='PANTHER',
            append_embed='none',
            mult=1,
            net_indiv=False,
            numOfproto=16,
            proto_path=None, 
            transformer_config=None):
        """
        The central co-attention module where you can do it all!

        Args:
            omic_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding
            num_classes: 4 if we are using NLL, 1 if we are using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space where histology and pathways are fused
            modality: ['gene','histo','coattn', 'partial'] 'coattn' accounts for both modalities
                If 'histo' or 'gene', unimodal self-attention
            histo_agg: ['mean', 'cat'] Take average of post-attention embeddings ('mean') or concatenate ('cat')
            histo_model: ['mil','PANTHER', 'OT', 'H2T']: 'mil' is for non-prototype-based methods
            net_indiv (bool): If True, create FFN for each prototype
            numOfproto: Number of histology prototypes
        """

        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.modality = modality

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes

        self.histo_model = histo_model.lower()

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()  # use this layer to calculate ig

        self.append_embed = append_embed

        if self.histo_model == 'panther':  # Uses prob/mean/cov
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))
        
        # 【修改点 1】保存基础维度，供 forward 中动态调整使用
        self.path_proj_dim_base = path_proj_dim 

        if self.histo_model != "mil":
            self.path_proj_dim, self.histo_embedding, self.gene_embedding = construct_proto_embedding(path_proj_dim,self.append_embed,self.numOfproto,len(omic_sizes))
            # 🔴 新增: 保存配置以便动态调整
            self.numOfproto_histo = numOfproto
            self.append_embed = append_embed
            # self.path_proj_dim = path_proj_dim
            self._embedding_initialized = False
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None


        coattn_list = []
        if self.num_coattn_layers == 0:
            out_dim = self.path_proj_dim

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   self.out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + len(omic_sizes))
            else:
                feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)
                
            if self.net_indiv:
                print("Warning: Disabling net_indiv because prototype count is dynamic. Using shared FeedForward.")
            feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))
            coattn_list.extend([feed_forward, layer_norm])
        else:
            out_dim = self.path_proj_dim // 2
            out_mult = self.out_mult
            
            if self.modality in ['histo', 'gene']: # If we want to use only single modality + self-attention
                attn_mode = 'self'
            elif self.modality == 'survpath':    # SurvPath setting H->P, P->H, P->P
                attn_mode = 'partial'
            else:
                attn_mode = 'full'  # Otherwise, perform self & cross attention

            cross_attender = MMAttentionLayer(
                dim=self.path_proj_dim,
                dim_head=out_dim,
                heads=1,
                residual=False,
                dropout=0.1,
                num_pathways=self.num_pathways,
                attn_mode=attn_mode
            )

            if self.net_indiv:  # Individual MLP per prototype
                feed_forward = FeedForwardEnsemble(out_dim,
                                                   out_mult,
                                                   dropout=dropout,
                                                   num=self.numOfproto + len(omic_sizes))
            else:
                feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)
            if self.net_indiv:
                print("Warning: Disabling net_indiv because prototype count is dynamic. Using shared FeedForward.")
            feed_forward = FeedForward(out_dim, out_mult, dropout=dropout)

            layer_norm = nn.LayerNorm(int(out_dim * out_mult))
            coattn_list.extend([cross_attender, feed_forward, layer_norm])


        self.coattn = nn.Sequential(*coattn_list)

        """
        # self.classifier = nn.Linear(288*3, self.num_classes, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(288 * 3, 128),  # 第一个全连接层
            nn.ReLU(),                # 激活函数
            nn.Linear(128, self.num_classes)  # 输出层
        )       

         # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(288)
             
        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(288)
        """
        if self.num_coattn_layers > 0:
            # Co-Attention 部分 (144 + 144) + Transformer 部分 (288 + 288) = 864
            # 公式: (path_proj_dim // 2) * 2 + path_proj_dim * 2 = path_proj_dim * 3
            clf_in_dim = self.path_proj_dim * 3
        else:
            # 如果没有 Co-Attention 层，维度保持不变
            # 288 * 4 = 1152
            clf_in_dim = self.path_proj_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, 128),  # 动态维度
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        
        # 加载Leiden信息
        leiden_info = None
        if proto_path is not None:
            try:
                from utils.file_utils import load_pkl

                print(f"[MultimodalModel] Loading proto from: {proto_path}")
                proto_data = load_pkl(proto_path)

                # 检查是否包含Leiden信息
                if 'feature_adjacency' in proto_data:
                    leiden_info = {
                        'n_proto': proto_data.get('n_proto'),
                        'feature_adjacency': proto_data.get('feature_adjacency'),
                        'spatial_centers': proto_data.get('spatial_centers'),
                        'spatial_adjacency': proto_data.get('spatial_adjacency'),
                        'spatial_spreads': proto_data.get('spatial_spreads'),
                    }
                    print(f"[MultimodalModel] ✓ Loaded Leiden info")
                    print(f"[MultimodalModel]   - n_proto: {leiden_info['n_proto']}")
                    print(f"[MultimodalModel]   - feature_adjacency: {leiden_info['feature_adjacency'].shape}")
                    print(f"[MultimodalModel]   - spatial_centers: {leiden_info['spatial_centers'].shape}")
                    # 🔥 打印spatial_spreads信息
                    if 'spatial_spreads' in leiden_info and leiden_info['spatial_spreads'] is not None:
                        spreads = leiden_info['spatial_spreads']
                        print(f"[MultimodalModel]   - spatial_spreads: {spreads.shape}, "
                              f"range=[{spreads.min():.2f}, {spreads.max():.2f}]")
                    else:
                        print(f"[MultimodalModel]   - spatial_spreads: NOT FOUND")
                else:
                    print(f"[MultimodalModel] Proto file has no Leiden info, using standard architecture")
            except Exception as e:
                print(f"[MultimodalModel] Failed to load Leiden info: {e}")
                import traceback
                traceback.print_exc()
                leiden_info = None
        else:
            print(f"[MultimodalModel] No proto_path specified, using standard architecture")

        # 创建Transformer（传入Leiden信息）
        # 创建Transformer_P（传入config）
        self.pathomics_encoder = Transformer_P(
            self.path_proj_dim,
            leiden_info=leiden_info,
            config=transformer_config  # ← 新增
        )
        self.genomics_encoder = Transformer_G(self.path_proj_dim)

    def forward_no_loss_after(self, x_path, x_omics, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:
        """
        device = x_path.device

        ## Pathway embeddings
        h_omic = []  ## each omic signature goes through it's own FC layer
        for idx, sig_feat in enumerate(x_omics):
            # 调试信息
            print(f"[Debug] idx={idx}, sig_feat.shape={sig_feat.shape}")

            # 维度修正
            if sig_feat.dim() == 1:
                print(f"[Warning] Fixing dimension at idx={idx}")
                sig_feat = sig_feat.unsqueeze(0)

            omic_feat = self.sig_networks[idx](sig_feat.float())
            print(f"[Debug] idx={idx}, omic_feat.shape={omic_feat.shape}")
            h_omic.append(omic_feat)

        print(f"[Debug] Before stack: len(h_omic)={len(h_omic)}")
        h_omic = torch.stack(h_omic, dim=1)
        print(f"[Debug] After stack: h_omic.shape={h_omic.shape}")
        
        if self.gene_embedding is not None: # Append gene prototype encoding
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            
            h_omic = torch.cat(arr, dim=0)
            # torch.Size([64, 50, 288])
            # print(h_omic.shape)
            # print("h_omic.shape")
        

        ## Histology embeddings
        # Project wsi to smaller dimension (same as pathway dimension)
        h_path = self.path_proj_net(x_path)
        # torch.Size([64, 16, 288])
        # print(h_path.shape)
        # print("h_path.shape")

        if self.histo_embedding is not None:    # Append histo prototype encoding
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)
            # torch.Size([64, 16, 288])
            # print(h_path.shape)
            # print("h_path.shape")

        
         # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            h_path)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            h_omic)  # cls token + patch tokens
        # torch.Size([64, 288])
        # print(patch_token_pathomics_encoder.shape)
        # print("patch_token_pathomics_encoder.shape")
        # # torch.Size([64, 288])
        # print(patch_token_genomics_encoder.shape)
        # print("patch_token_genomics_encoder")

        

        # # # 拼接交互融合
        tokens = torch.cat([patch_token_genomics_encoder, patch_token_pathomics_encoder], dim=1) # (B, N_p+N_h, d)
        tokens = self.identity(tokens)

        # # # # Required for visualization
        if return_attn:
            with torch.no_grad():
                _, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn[0](x=tokens, mask=None, return_attention=True)

        # # # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)
        # # torch.Size([64, 347, 144])
        # # # print(mm_embed.shape)
        # # # print("mm_embed.shape")


        # # ---> aggregate
        # # Pathways
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        # Histology
        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)
   
       


        if self.modality == 'histo':    # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':   # Just use gene for prediction
            embedding = paths_postSA_embed
        else:   # Use both modalities
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed,cls_token_pathomics_encoder,cls_token_genomics_encoder], dim=1)  # ---> both branches

        # print(embedding.shape)
        # print("embedding.shape")
        logits = self.classifier(embedding)
        out = {'logits': logits}
        if return_attn:
            out['omic_attn'] = attn_pathways
            print(attn_pathways.shape)
            out['cross_attn'] = cross_attn_pathways
            print(cross_attn_pathways.shape)
            out['path_attn'] = cross_attn_histology
            print(cross_attn_histology.shape)

        return out
     
       
    def forward_no_loss(self, x_path, x_omics, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:

        """
        device = x_path.device
        
        # 🔴 新增: 动态调整embedding维度
        if self.histo_embedding is not None:
            actual_numOfproto = x_path.shape[1]
            
            if actual_numOfproto != self.numOfproto_histo or not self._embedding_initialized:
                print(f"🔧 Adjusting histo_embedding: {self.numOfproto_histo} → {actual_numOfproto}")
                
                _, new_histo_emb, _ = construct_proto_embedding(
                    self.path_proj_dim,
                    self.append_embed,
                    actual_numOfproto,
                    len(self.sig_networks)
                )
                
                if isinstance(new_histo_emb, torch.nn.Parameter):
                    # 已经是Parameter,直接注册
                    self.register_parameter('histo_embedding', torch.nn.Parameter(new_histo_emb.to(device)))
                else:
                    # 普通Tensor,转为Parameter后注册
                    self.register_parameter('histo_embedding', torch.nn.Parameter(new_histo_emb.to(device), requires_grad=True))
                
                self.numOfproto_histo = actual_numOfproto
                self._embedding_initialized = True
        
        ## Pathway embeddings
        h_omic = []  ## each omic signature goes through it's own FC layer
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())  # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)

        if self.gene_embedding is not None: # Append gene prototype encoding
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            
            h_omic = torch.cat(arr, dim=0)
            # torch.Size([64, 50, 288])
            # print(h_omic.shape)
            # print("h_omic.shape")
        

        ## Histology embeddings
        # Project wsi to smaller dimension (same as pathway dimension)
        h_path = self.path_proj_net(x_path)
        # torch.Size([64, 16, 288])
        # print(h_path.shape)
        # print("h_path.shape")

        if self.histo_embedding is not None:    # Append histo prototype encoding
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)
            # torch.Size([64, 16, 288])
            # print(h_path.shape)
            # print("h_path.shape")

        
         # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            h_path)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            h_omic)  # cls token + patch tokens
        # torch.Size([64, 288])
        # print(patch_token_pathomics_encoder.shape)
        # print("patch_token_pathomics_encoder.shape")
        # # torch.Size([64, 288])
        # print(patch_token_genomics_encoder.shape)
        # print("patch_token_genomics_encoder")

        

        # # # 拼接交互融合
        tokens = torch.cat([patch_token_genomics_encoder, patch_token_pathomics_encoder], dim=1) # (B, N_p+N_h, d)
        tokens = self.identity(tokens)

        # # # # Required for visualization
        if return_attn:
            with torch.no_grad():
                _, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn[0](x=tokens, mask=None, return_attention=True)

        # # # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)
        # # torch.Size([64, 347, 144])
        # # # print(mm_embed.shape)
        # # # print("mm_embed.shape")


        # # ---> aggregate
        # # Pathways
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        # Histology
        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)
   
       


        if self.modality == 'histo':    # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':   # Just use gene for prediction
            embedding = paths_postSA_embed
        else:   # Use both modalities
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed,cls_token_pathomics_encoder,cls_token_genomics_encoder], dim=1)  # ---> both branches

        # print(embedding.shape)
        # print("embedding.shape")
        logits = self.classifier(embedding)
        out = {'logits': logits}
        if return_attn:
            out['omic_attn'] = attn_pathways
            print(attn_pathways.shape)
            out['cross_attn'] = cross_attn_pathways
            print(cross_attn_pathways.shape)
            out['path_attn'] = cross_attn_histology
            print(cross_attn_histology.shape)

        return out

    def forward(self, x_path, x_omics, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):

        out = self.forward_no_loss(x_path, x_omics, return_attn)

        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        if return_attn:
            results_dict['omic_attn'] = out['omic_attn']
            results_dict['cross_attn'] = out['cross_attn']
            results_dict['path_attn'] = out['path_attn']

        results_dict.update(out)
        return results_dict, log_dict



class coattn_mot(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            histo_in_dim=1024,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256,
            num_coattn_layers=1,
            modality='both',
            histo_agg='mean',
            histo_model='mil',
            append_embed='none',
            mult=1,
            net_indiv=False,
            numOfproto=16,
            ot_reg=0.1,
            ot_tau=0.5,
            ot_impl="pot-uot-l2"):
        """
        Multimodal fusion with cross-modal optimal transport (instead of Transformer cross-attention).
        Visualization currently not supported

        Args:
            omic_sizes: List of integers, each indicating number of genes per prototype
            histo_in_dim: Dimension of histology feature embedding
            num_classes: 4 if we are using NLL, 1 if we are using Cox/Ranking loss
            path_proj_dim: Dimension of the embedding space where histology and pathways are fused
            modality: ['gene','histo','coattn'] 'coattn' accounts for both modalities
            histo_agg: ['mean', 'cat'] Take average of post-attention embeddings ('mean') or concatenate ('cat')
            histo_model: ['mil','PANTHER', 'OT', 'H2T']: 'mil' is for non-prototype-based methods
            net_indiv (bool): If True, create FFN for each prototype
            numOfproto: Number of histology prototypes
        """

        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.num_coattn_layers = num_coattn_layers

        self.histo_in_dim = histo_in_dim
        self.out_mult = mult
        self.net_indiv = net_indiv
        self.modality = modality

        self.histo_agg = histo_agg

        self.numOfproto = numOfproto
        self.num_classes = num_classes

        self.histo_model = histo_model.lower()

        if self.histo_model == 'panther':  # Uses prob/mean/cov
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim * 2 + 1, path_proj_dim))
        else:
            self.path_proj_net = nn.Sequential(nn.Linear(self.histo_in_dim, path_proj_dim))

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()  # use this layer to calculate ig

        self.append_embed = append_embed

        if histo_model != "mil":
            self.path_proj_dim, self.histo_embedding, self.gene_embedding = construct_proto_embedding(path_proj_dim,
                                                                                                      self.append_embed,
                                                                                                      self.numOfproto,
                                                                                                      len(omic_sizes))
        else:
            self.path_proj_dim = path_proj_dim
            self.histo_embedding = None
            self.gene_embedding = None       


        ### OT-based Co-attention
        out_dim = self.path_proj_dim // 2
        self.out_mult = mult
        self.mot = OT_Attn(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

        coattn = MMAttentionLayer(
                                    dim=self.path_proj_dim,
                                    dim_head=out_dim,
                                    heads=1,
                                    residual=False,
                                    dropout=0.1,
                                    num_pathways=self.num_pathways,
                                    attn_mode='self'
                                )

        if self.net_indiv:  # Individual MLP per prototype
            feed_forward = FeedForwardEnsemble(out_dim,
                                               self.out_mult,
                                               dropout=dropout,
                                               num=self.numOfproto + len(omic_sizes))
        else:
            feed_forward = FeedForward(out_dim, self.out_mult, dropout=dropout)

        layer_norm = nn.LayerNorm(int(out_dim * self.out_mult))

        self.coattn = nn.Sequential(coattn, feed_forward, layer_norm)

        out_dim_final = int(out_dim * self.out_mult)
        histo_final_dim = out_dim_final * self.numOfproto if self.histo_agg == 'cat' else out_dim_final
        gene_final_dim = out_dim_final

        if self.modality == 'histo':
            in_dim = histo_final_dim
        elif self.modality == 'gene':
            in_dim = gene_final_dim
        else:
            in_dim = histo_final_dim + gene_final_dim

        self.classifier = nn.Linear(in_dim, self.num_classes, bias=False)


    def forward_no_loss(self, x_path, x_omics, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:

        Returns:

        """
        device = x_path.device

        ## Pathway embeddings
        h_omic = []  ## each omic signature goes through it's own FC layer
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float())  # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)

        if self.gene_embedding is not None: # Append gene prototype encoding
            arr = []
            for idx in range(len(h_omic)):
                arr.append(torch.cat([h_omic[idx:idx + 1], self.gene_embedding.to(device)], dim=-1))
            h_omic = torch.cat(arr, dim=0)

        ## Histology embeddings
        # Project wsi to smaller dimension (same as pathway dimension)
        h_path = self.path_proj_net(x_path)

        if self.histo_embedding is not None:    # Append histo prototype encoding
            arr = []
            for idx in range(len(h_path)):
                arr.append(torch.cat([h_path[idx:idx + 1], self.histo_embedding.to(device)], dim=-1))
            h_path = torch.cat(arr, dim=0)

        #
        # Cross-modal optimal transport
        #
        p_to_h_coattn = self.mot(h_path, h_omic).float()
        h_to_p_coattn = self.mot(h_omic, h_path).float()

        h_path_coattn = torch.matmul(p_to_h_coattn, h_path)
        h_omic_coattn = torch.matmul(h_to_p_coattn, h_omic)

        tokens = torch.cat([h_path_coattn, h_omic_coattn], dim=1)  # (B, N_p+N_h, d)
        tokens = self.identity(tokens)

        # Pass the token set through co-attention network
        mm_embed = self.coattn(tokens)

        # ---> aggregate
        # modality specific mean
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]

        if self.histo_model == 'mil':
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # For non-prototypes, we just take the mean
        else:
            wsi_postSA_embed = agg_histo(wsi_postSA_embed, self.histo_agg)

        if self.modality == 'histo':    # Just use histo for prediction
            embedding = wsi_postSA_embed
        elif self.modality == 'gene':   # Just use gene for prediction
            embedding = paths_postSA_embed
        else:   # Use both modalities
            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)  # ---> both branches

        logits = self.classifier(embedding)
        out = {'logits': logits}

        return out

    def forward(self, x_path, x_omics, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):

        out = self.forward_no_loss(x_path, x_omics, return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        results_dict.update(out)

        return results_dict, log_dict



class SurvPath(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            path_dim=1024,
            dropout=0.1,
            num_classes=4,
            path_proj_dim=256):
        """
        SurvPath model. Uses P->P, H->P, P->H interaction model
        """

        super().__init__()

        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.path_dim = path_dim
        self.path_proj_dim = path_proj_dim

        self.num_classes = num_classes
        self.path_proj_net = nn.Sequential(nn.Linear(self.path_dim, self.path_proj_dim))

        self.sig_networks = init_per_path_model(omic_sizes)
        self.identity = nn.Identity()  # use this layer to calculate ig

        out_dim = self.path_proj_dim // 2
        cross_attender = MMAttentionLayer(
            dim=self.path_proj_dim,
            dim_head=out_dim,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_pathways,
            attn_mode='partial'
        )

        feed_forward = FeedForward(out_dim, dropout=dropout)
        layer_norm = nn.LayerNorm(out_dim)

        self.coattn = nn.Sequential(*[cross_attender, feed_forward, layer_norm])

        histo_final_dim = self.path_proj_dim // 2
        gene_final_dim = self.path_proj_dim // 2

        self.classifier = nn.Linear(histo_final_dim + gene_final_dim, self.num_classes)

    def forward_no_loss(self, x_path, x_omics, return_attn=False):
        """
        Args:
            x_path: (B, numOfproto, in_dim) in_dim = [prob, mean, cov] (If OT, prob will be uniform, cov will be none)
            x_omics:
            return_attn:

        Returns:

        """
        mask = None

        x_path_mean = x_path


        # ---> get pathway embeddings
        h_omic = [] ## each omic signature goes through it's own FC layer
        for idx, sig_feat in enumerate(x_omics):
            omic_feat = self.sig_networks[idx](sig_feat.float()) # (B, d)
            h_omic.append(omic_feat)
        h_omic = torch.stack(h_omic, dim=1)

        # ---> project wsi to smaller dimension (same as pathway dimension)
        h_path = self.path_proj_net(x_path_mean)

        tokens = torch.cat([h_omic, h_path], dim=1) # (B, N_p+N_h, d)
        tokens = self.identity(tokens)

       
        # if return_attn:
        #     mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.coattn(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        # else:
        #     mm_embed = self.coattn(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        mm_embed = self.coattn(tokens)

        # ---> aggregate
        # modality specific mean
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)  # ---> both branches

        logits = self.classifier(embedding)

        out = {'logits': logits}

        return out

    def forward(self, x_path, x_omics, return_attn=False, attn_mask=None, label=None, censorship=None, loss_fn=None):
        out = self.forward_no_loss(x_path, x_omics, return_attn)
        results_dict, log_dict = process_surv(out['logits'], label, censorship, loss_fn)
        results_dict.update(out)

        return results_dict, log_dict
