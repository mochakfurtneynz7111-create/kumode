
### DMSS model_PANTHER.py
# Model initiation for PANTHER

from torch import nn
import numpy as np

from .components import predict_surv, predict_clf, predict_emb
from .PANTHER.layers import PANTHERBase
from utils.proto_utils import check_prototypes


class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, config, mode):
        super(PANTHER, self).__init__()

        self.config = config
        emb_dim = config.in_dim
        
        # === 关键修改: 如果使用Leiden,从原型文件读取实际数量 ===
        if config.load_proto and hasattr(config, 'use_leiden') and config.use_leiden:
            # 从原型文件读取实际的聚类数
            from utils.file_utils import load_pkl
            proto_data = load_pkl(config.proto_path)
            
            if 'n_proto' in proto_data:
                actual_n_proto = proto_data['n_proto']
                print(f"[PANTHER] Loading Leiden prototypes: {actual_n_proto} prototypes")
                config.out_size = actual_n_proto  # 动态更新!
            else:
                # 兼容旧格式
                actual_n_proto = proto_data['prototypes'].shape[1]
                config.out_size = actual_n_proto
        
        self.emb_dim = emb_dim
        self.heads = config.heads
        self.outsize = config.out_size  # 现在是动态的!
        self.load_proto = config.load_proto
        self.mode = mode

        check_prototypes(config.out_size, self.emb_dim, self.load_proto, config.proto_path)
        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, p=config.out_size, L=config.em_iter,
                         tau=config.tau, out=config.out_type, ot_eps=config.ot_eps,
                         load_proto=config.load_proto, proto_path=config.proto_path,
                         fix_proto=config.fix_proto)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out, qqs = self.panther(x)
        
        # ❌ 删除这些行:
        # qq = qqs[0,:,:,0].cpu().numpy()
        # global_cluster_labels = qq.argmax(axis=1)
        
        # TODO 各个类别概率
        # np.set_printoptions(threshold=np.inf)  # 打印所有元素
        # print(global_cluster_labels)
        


        return {'repr': out, 'qq': qqs}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y
