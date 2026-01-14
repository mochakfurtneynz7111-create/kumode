import pickle
import numpy as np
import os
import h5py
import openslide
import torch
import pandas as pd
import sys
import matplotlib
# 强制使用非交互式后端，防止在服务器终端运行时报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 假设这些工具类在当前目录或 PYTHONPATH 中
sys.path.append('../')
from prototype_visualization_utils import get_panther_encoder, visualize_categorical_heatmap, get_mixture_plot, get_default_cmap
from mmp_visualization_utils import plot_pathomic_correspondence
from mil_models.tokenizer import PrototypeTokenizer

# ================= 配置路径 =================
# 请确保这些路径在你的服务器上是正确的
import pickle
import numpy as np
import os
import h5py
import openslide
import torch
import pandas as pd
import sys
import matplotlib
# 强制使用非交互式后端，防止在服务器终端运行时报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 假设这些工具类在当前目录或 PYTHONPATH 中
sys.path.append('../')
from prototype_visualization_utils import get_panther_encoder, visualize_categorical_heatmap, get_mixture_plot, get_default_cmap
from mmp_visualization_utils import plot_pathomic_correspondence
from mil_models.tokenizer import PrototypeTokenizer

# ================= 配置路径 =================
# 请确保这些路径在你的服务器上是正确的
proto_path = "/root/autodl-tmp/DMSS/DMSS/src/splits/survival_UNI/TCGA_KIRC_overall_survival_k=3/prototypes/prototypes_c20_extracted-vit_large_patch16_224.dinov2.uni_mass100k_leiden_res0.1_num_1.0e+05.pkl"
mmp_result_pkl_path = "/root/autodl-tmp/DMSS/DMSS/src/results/KIRC_survival::PANTHER_default_leiden_G_P_p20::extracted-vit_large_patch16_224.dinov2.uni_mass100k/KIRC_survival/k=3/TCGA_KIRC_overall_survival::PANTHER_default::feats_h5/TCGA_KIRC_overall_survival::PANTHER_default::feats_h5::26-01-12-23-22-33/test_results.pkl"

# WSI 和 特征路径
slide_id = "TCGA-BP-4163-01Z-00-DX1.1dc1c4fb-2691-42f8-b62f-c51db47b30dc"
slide_fpath = f'/root/autodl-tmp/DMSS/DMSS/src/data/svs/tcga_kirc/{slide_id}.svs'
h5_feats_fpath = f'/root/autodl-tmp/UNI2-h/UNI2_features/TCGA/ALL_KIRC_embeddings/extracted_mag20x_patch256_fp/extracted-vit_large_patch16_224.dinov2.uni_mass100k/feats_h5/{slide_id}.h5'

# Hallmarks CSV 路径 (请修改此处)
hallmarks_csv_path = '/root/autodl-tmp/DMSS/DMSS/src/data_csvs/rna/metadata/hallmarks_signatures.csv' 

# ================= 设置保存目录 =================
sample_ids = ["TCGA-BP-4163"]
sample_id = sample_ids[0]

# 创建保存结果的主文件夹
save_dir = f'./visualization_results/{sample_id}'
crossmodal_dir = os.path.join(save_dir, 'crossmodal')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(crossmodal_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(save_dir)}")

# ================= 加载模型与数据 =================
print("Loading PANTHER Encoder...")
panther_encoder = get_panther_encoder(in_dim=1024, p=16, proto_path=proto_path, config_dir='../configs')
panther_encoder.panther.H = 1

print("Loading MMP Results...")
results = pickle.load(open(mmp_result_pkl_path, 'rb'))
for split, split_results in results.items():
    print(f"Split loaded: {split}")

idxs = [np.where(results['sample_ids']==sample_id)[0][0] for sample_id in sample_ids]
sampleid2idx = dict(zip(sample_ids, idxs))

# ================= Color Map 设置 =================
color_map = get_default_cmap(16)
for k,v in get_default_cmap(16).items():
    color_map[15-k] = v
color_map_hex = get_default_cmap(16, return_hex=True)
for k,v in get_default_cmap(16, return_hex=True).items():
    color_map_hex[15-k] = v

# 保存 colormap 信息到文本文件，代替 display
with open(os.path.join(save_dir, 'colormap_info.txt'), 'w') as f:
    f.write(str(color_map) + '\n')
    f.write(str(color_map_hex) + '\n')

# ================= 处理 WSI 和 特征 =================
print(f"Processing slide: {slide_id}")
wsi = openslide.open_slide(slide_fpath)
h5 = h5py.File(h5_feats_fpath, 'r')

coords = h5['coords'][:] 
feats = torch.Tensor(h5['features'][:])
custom_downsample = 2
patch_size = 256

# ================= PANTHER 推理 =================
print("Running inference...")
with torch.inference_mode():
    out, qqs = panther_encoder.representation(feats.unsqueeze(dim=0)).values()
    tokenizer = PrototypeTokenizer(p=16, out_type='allcat')
    
    # 打印一些调试信息
    # print(out.shape)
    
    mus, pis, sigmas = tokenizer.forward(out)
    mus = mus[0].detach().cpu().numpy()

    qq = qqs[0,:,:,0].cpu().numpy()
    global_cluster_labels = qq.argmax(axis=1)
    
    print("========== Cluster Statistics ==========")
    from collections import Counter
    counter = Counter(global_cluster_labels)
    total_count = sum(counter.values())
    proportions = {key: count / total_count for key, count in counter.items()}
    
    stat_path = os.path.join(save_dir, 'cluster_stats.txt')
    with open(stat_path, 'w') as f:
        for key, count in counter.items():
            line = f"Class: {key}, Count: {count}, Ratio: {proportions[key]:.2%}"
            print(line)
            f.write(line + '\n')

# ================= 可视化：Categorical Heatmap =================
print("Generating Categorical Heatmap...")
cat_map = visualize_categorical_heatmap(
    wsi,
    coords, 
    global_cluster_labels, 
    label2color_dict=color_map,
    vis_level=wsi.get_best_level_for_downsample(128),
    patch_size=(patch_size, patch_size),
    alpha=0.4,
)

# 保存 Heatmap (PIL Image)
heatmap_save_path = os.path.join(save_dir, 'categorical_heatmap.png')
cat_map_resized = cat_map.resize((cat_map.width//4, cat_map.height//4))
cat_map_resized.save(heatmap_save_path)
print(f"Saved heatmap to {heatmap_save_path}")

# ================= 可视化：GMM Mixtures =================
print("Generating GMM Mixture Plot...")
# 获取 matplotlib figure 对象
mixture_fig = get_mixture_plot(mus, colors=list(color_map_hex.values()))

# 保存 Mixture Plot
mixture_save_path = os.path.join(save_dir, 'gmm_mixture.png')
# 检查 mixture_fig 是 Figure 还是 Axes，通常是 Figure，如果是 Figure 直接保存
if hasattr(mixture_fig, 'savefig'):
    mixture_fig.savefig(mixture_save_path, bbox_inches='tight')
else:
    # 如果返回的是 plot 对象或其他，尝试获取 figure
    try:
        mixture_fig.figure.savefig(mixture_save_path, bbox_inches='tight')
    except:
        print("Warning: Could not save mixture plot directly. Check return type of get_mixture_plot.")
plt.close('all') # 释放内存

# ================= 加载 Hallmarks 和 Cross-Attention =================
# 检查 CSV 是否存在
if not os.path.exists(hallmarks_csv_path):
    print(f"Warning: Hallmarks CSV not found at {hallmarks_csv_path}. Skipping attention plots.")
else:
    hallmarks = pd.read_csv(hallmarks_csv_path)
    hallmarks = sorted([' '.join(x[9:].split('_')) for x in hallmarks.columns])
    
    cluster_ids = [f'C{i}' for i in range (16)]
    # 这里定义了类别名称
    cluster_identities_list = ['Tumor', 'Adipose/Connective', 'Connective', 'Connective', 'Adipose', 'Immune/Normal', 'Adipose/Tumor', 'Tumor', 'Tumor', 'Dense Tumor', 'Aritfacts', 'Normal Ducts/FEA', 'Dense Tumor', 'Tumor/Fat/Stroma', 'Normal', 'Tumor/Fat']
    cluster_identities = dict(zip(cluster_ids, [x+' '+y for x,y in zip(cluster_ids, cluster_identities_list)]))

    idx = sampleid2idx[sample_id]

    # Path -> Omic Attention
    cross_attn_path2omic = results['test']['all_cross_attn'][idx]
    cross_attn_path2omic = torch.nn.Softmax(dim=1)(torch.Tensor(cross_attn_path2omic)).cpu().numpy()
    cross_attn_path2omic = pd.DataFrame(cross_attn_path2omic, index=hallmarks, columns=cluster_identities.values())
    print('cross_attn_path2omic shape:', cross_attn_path2omic.shape)

    # Omic -> Path Attention
    cross_attn_omic2path = results['test']['all_path_attn'][idx]
    cross_attn_omic2path = torch.nn.Softmax(dim=1)(torch.Tensor(cross_attn_omic2path)).cpu().numpy()
    cross_attn_omic2path = pd.DataFrame(cross_attn_omic2path, columns=hallmarks, index=cluster_ids)
    print('cross_attn_omic2path shape:', cross_attn_omic2path.shape)

    # ================= 可视化并保存：Omic-to-Path Interactions =================
    print("Saving Omic-to-Path visualizations...")
    for cluster in cross_attn_omic2path.index:
        omic2path = cross_attn_omic2path.loc[cluster].sort_values(ascending=False)
        cluster_name = cluster_identities[cluster]
        
        # 生成图表
        fig = plot_pathomic_correspondence(
            omic2path, orient='v', color='#e7727a', 
            axis_lim=[0.0, 0.081], axis_tick=[0.0, 0.02, 0.04, 0.06, 0.08], lim=10)
        
        # 格式化文件名 (移除斜杠等非法字符)
        safe_name = cluster_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(crossmodal_dir, f'omic2path_{safe_name}.png')
        
        if hasattr(fig, 'savefig'):
            fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    # ================= 可视化并保存：Path-to-Omic Interactions =================
    print("Saving Path-to-Omic visualizations...")
    for idx, omic in enumerate(cross_attn_path2omic.index):
        path2omic = cross_attn_path2omic.loc[omic].sort_values(ascending=False)
        
        # 生成图表
        fig = plot_pathomic_correspondence(path2omic, color='#8693e8')
        
        safe_name = "_".join(omic.lower().split(" "))
        save_path = os.path.join(crossmodal_dir, f'path2omic_hallmark{idx}_{safe_name}.png')
        
        if hasattr(fig, 'savefig'):
            fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

print("All visualizations completed and saved.")
mmp_result_pkl_path = "/home/zhuzhuwen/code/MMP_NETWORK/src/results/KIRC_survival::PANTHER_default::vit_large_patch16_224.dinov2.uni_mass100k/KIRC_survival/k=3/TCGA_KIRC_overall_survival::PANTHER_default::feats_h5/TCGA_KIRC_overall_survival::PANTHER_default::feats_h5::24-12-22-16-15-29/test_results.pkl"

# WSI 和 特征路径
slide_id = "TCGA-BP-4163-01Z-00-DX1.1dc1c4fb-2691-42f8-b62f-c51db47b30dc"
slide_fpath = f'/home/zhuzhuwen/data/svs/tcga_kirc/{slide_id}.svs'
h5_feats_fpath = f'/home/zhuzhuwen/data/features/uni/tcga_kirc/extracted_mag20x_patch256_fp/extracted-vit_large_patch16_224.dinov2.uni_mass100k/feats_h5/{slide_id}.h5'

# Hallmarks CSV 路径 (请修改此处)
hallmarks_csv_path = '<path/to//hallmarks_signatures.csv>' 

# ================= 设置保存目录 =================
sample_ids = ["TCGA-BP-4163"]
sample_id = sample_ids[0]

# 创建保存结果的主文件夹
save_dir = f'./visualization_results/{sample_id}'
crossmodal_dir = os.path.join(save_dir, 'crossmodal')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(crossmodal_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(save_dir)}")

# ================= 加载模型与数据 =================
print("Loading PANTHER Encoder...")
panther_encoder = get_panther_encoder(in_dim=1024, p=16, proto_path=proto_path, config_dir='../configs')
panther_encoder.panther.H = 1

print("Loading MMP Results...")
results = pickle.load(open(mmp_result_pkl_path, 'rb'))
for split, split_results in results.items():
    print(f"Split loaded: {split}")

idxs = [np.where(results['sample_ids']==sample_id)[0][0] for sample_id in sample_ids]
sampleid2idx = dict(zip(sample_ids, idxs))

# ================= Color Map 设置 =================
color_map = get_default_cmap(16)
for k,v in get_default_cmap(16).items():
    color_map[15-k] = v
color_map_hex = get_default_cmap(16, return_hex=True)
for k,v in get_default_cmap(16, return_hex=True).items():
    color_map_hex[15-k] = v

# 保存 colormap 信息到文本文件，代替 display
with open(os.path.join(save_dir, 'colormap_info.txt'), 'w') as f:
    f.write(str(color_map) + '\n')
    f.write(str(color_map_hex) + '\n')

# ================= 处理 WSI 和 特征 =================
print(f"Processing slide: {slide_id}")
wsi = openslide.open_slide(slide_fpath)
h5 = h5py.File(h5_feats_fpath, 'r')

coords = h5['coords'][:] 
feats = torch.Tensor(h5['features'][:])
custom_downsample = 2
patch_size = 256

# ================= PANTHER 推理 =================
print("Running inference...")
with torch.inference_mode():
    out, qqs = panther_encoder.representation(feats.unsqueeze(dim=0)).values()
    tokenizer = PrototypeTokenizer(p=16, out_type='allcat')
    
    # 打印一些调试信息
    # print(out.shape)
    
    mus, pis, sigmas = tokenizer.forward(out)
    mus = mus[0].detach().cpu().numpy()

    qq = qqs[0,:,:,0].cpu().numpy()
    global_cluster_labels = qq.argmax(axis=1)
    
    print("========== Cluster Statistics ==========")
    from collections import Counter
    counter = Counter(global_cluster_labels)
    total_count = sum(counter.values())
    proportions = {key: count / total_count for key, count in counter.items()}
    
    stat_path = os.path.join(save_dir, 'cluster_stats.txt')
    with open(stat_path, 'w') as f:
        for key, count in counter.items():
            line = f"Class: {key}, Count: {count}, Ratio: {proportions[key]:.2%}"
            print(line)
            f.write(line + '\n')

# ================= 可视化：Categorical Heatmap =================
print("Generating Categorical Heatmap...")
cat_map = visualize_categorical_heatmap(
    wsi,
    coords, 
    global_cluster_labels, 
    label2color_dict=color_map,
    vis_level=wsi.get_best_level_for_downsample(128),
    patch_size=(patch_size, patch_size),
    alpha=0.4,
)

# 保存 Heatmap (PIL Image)
heatmap_save_path = os.path.join(save_dir, 'categorical_heatmap.png')
cat_map_resized = cat_map.resize((cat_map.width//4, cat_map.height//4))
cat_map_resized.save(heatmap_save_path)
print(f"Saved heatmap to {heatmap_save_path}")

# ================= 可视化：GMM Mixtures =================
print("Generating GMM Mixture Plot...")
# 获取 matplotlib figure 对象
mixture_fig = get_mixture_plot(mus, colors=list(color_map_hex.values()))

# 保存 Mixture Plot
mixture_save_path = os.path.join(save_dir, 'gmm_mixture.png')
# 检查 mixture_fig 是 Figure 还是 Axes，通常是 Figure，如果是 Figure 直接保存
if hasattr(mixture_fig, 'savefig'):
    mixture_fig.savefig(mixture_save_path, bbox_inches='tight')
else:
    # 如果返回的是 plot 对象或其他，尝试获取 figure
    try:
        mixture_fig.figure.savefig(mixture_save_path, bbox_inches='tight')
    except:
        print("Warning: Could not save mixture plot directly. Check return type of get_mixture_plot.")
plt.close('all') # 释放内存

# ================= 加载 Hallmarks 和 Cross-Attention =================
# 检查 CSV 是否存在
if not os.path.exists(hallmarks_csv_path):
    print(f"Warning: Hallmarks CSV not found at {hallmarks_csv_path}. Skipping attention plots.")
else:
    hallmarks = pd.read_csv(hallmarks_csv_path)
    hallmarks = sorted([' '.join(x[9:].split('_')) for x in hallmarks.columns])
    
    cluster_ids = [f'C{i}' for i in range (16)]
    # 这里定义了类别名称
    cluster_identities_list = ['Tumor', 'Adipose/Connective', 'Connective', 'Connective', 'Adipose', 'Immune/Normal', 'Adipose/Tumor', 'Tumor', 'Tumor', 'Dense Tumor', 'Aritfacts', 'Normal Ducts/FEA', 'Dense Tumor', 'Tumor/Fat/Stroma', 'Normal', 'Tumor/Fat']
    cluster_identities = dict(zip(cluster_ids, [x+' '+y for x,y in zip(cluster_ids, cluster_identities_list)]))

    idx = sampleid2idx[sample_id]

    # Path -> Omic Attention
    cross_attn_path2omic = results['test']['all_cross_attn'][idx]
    cross_attn_path2omic = torch.nn.Softmax(dim=1)(torch.Tensor(cross_attn_path2omic)).cpu().numpy()
    cross_attn_path2omic = pd.DataFrame(cross_attn_path2omic, index=hallmarks, columns=cluster_identities.values())
    print('cross_attn_path2omic shape:', cross_attn_path2omic.shape)

    # Omic -> Path Attention
    cross_attn_omic2path = results['test']['all_path_attn'][idx]
    cross_attn_omic2path = torch.nn.Softmax(dim=1)(torch.Tensor(cross_attn_omic2path)).cpu().numpy()
    cross_attn_omic2path = pd.DataFrame(cross_attn_omic2path, columns=hallmarks, index=cluster_ids)
    print('cross_attn_omic2path shape:', cross_attn_omic2path.shape)

    # ================= 可视化并保存：Omic-to-Path Interactions =================
    print("Saving Omic-to-Path visualizations...")
    for cluster in cross_attn_omic2path.index:
        omic2path = cross_attn_omic2path.loc[cluster].sort_values(ascending=False)
        cluster_name = cluster_identities[cluster]
        
        # 生成图表
        fig = plot_pathomic_correspondence(
            omic2path, orient='v', color='#e7727a', 
            axis_lim=[0.0, 0.081], axis_tick=[0.0, 0.02, 0.04, 0.06, 0.08], lim=10)
        
        # 格式化文件名 (移除斜杠等非法字符)
        safe_name = cluster_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(crossmodal_dir, f'omic2path_{safe_name}.png')
        
        if hasattr(fig, 'savefig'):
            fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    # ================= 可视化并保存：Path-to-Omic Interactions =================
    print("Saving Path-to-Omic visualizations...")
    for idx, omic in enumerate(cross_attn_path2omic.index):
        path2omic = cross_attn_path2omic.loc[omic].sort_values(ascending=False)
        
        # 生成图表
        fig = plot_pathomic_correspondence(path2omic, color='#8693e8')
        
        safe_name = "_".join(omic.lower().split(" "))
        save_path = os.path.join(crossmodal_dir, f'path2omic_hallmark{idx}_{safe_name}.png')
        
        if hasattr(fig, 'savefig'):
            fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

print("All visualizations completed and saved.")