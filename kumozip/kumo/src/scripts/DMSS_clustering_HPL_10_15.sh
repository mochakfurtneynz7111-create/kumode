#!/bin/bash
# DMSS_blca_prototype.sh - 修正版

gpuid=$1
split_dir=$2
split_names=$3

# 特征相关参数
feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1536
mag='20x'
patch_size=256

# 聚类参数
mode='leiden'  # 'faiss' or 'kmeans' or 'leiden'
n_sampling_patches=100000
n_init=3

# ========== Leiden特有参数 ==========
leiden_resolution=1.0  # 控制聚类数量: 0.5-2.0
leiden_neighbors=15

# ========== K-means/FAISS参数 (Leiden模式会忽略) ==========
n_proto=16  # 只在mode=kmeans/faiss时使用

# 数据路径处理
dataroots=("${@:4}")  # 从第4个参数开始是数据路径
all_feat_dirs=""
for dataroot_path in "${dataroots[@]}"; do
  feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir
  then
    continue
  fi
  if [[ -z ${all_feat_dirs} ]]; then
    all_feat_dirs=${feat_dir}
  else
    all_feat_dirs=${all_feat_dirs},${feat_dir}
  fi
done

# ========== 构建Python命令 ==========
if [ "$mode" == "leiden" ]; then
    # Leiden模式
    echo "==========================================="
    echo "Running Leiden clustering (HPL method)"
    echo "Resolution: ${leiden_resolution}"
    echo "Neighbors: ${leiden_neighbors}"
    echo "==========================================="
    
    cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_prototype \
--mode leiden \
--data_source ${all_feat_dirs} \
--split_dir ${split_dir} \
--split_names ${split_names} \
--in_dim ${input_dim} \
--n_proto_patches ${n_sampling_patches} \
--leiden_resolution ${leiden_resolution} \
--leiden_neighbors ${leiden_neighbors} \
--n_init ${n_init} \
--seed 1 \
--num_workers 10"

else
    # K-means/FAISS模式
    echo "==========================================="
    echo "Running ${mode} clustering"
    echo "Number of prototypes: ${n_proto}"
    echo "==========================================="
    
    cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_prototype \
--mode ${mode} \
--n_proto ${n_proto} \
--data_source ${all_feat_dirs} \
--split_dir ${split_dir} \
--split_names ${split_names} \
--in_dim ${input_dim} \
--n_proto_patches ${n_sampling_patches} \
--n_init ${n_init} \
--seed 1 \
--num_workers 10"
fi

# 打印并执行命令
echo ""
echo "Executing command:"
echo "$cmd"
echo ""

eval "$cmd"