#!/bin/bash
# ============================================================================
# DMSS Complete Configuration Script - 支持Leiden原型
# ============================================================================

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("${@:6}")  # 从第6个参数开始是数据路径

# ============================================================================
# 【1】数据和特征提取参数
# ============================================================================
feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1536
mag='20x'
patch_size=256

# ============================================================================
# 【2】训练基础参数
# ============================================================================
max_epoch=200
lr=0.0001
wd=0.00001
lr_scheduler='cosine'
opt='adamW'
grad_accum=1
batch_size=64
seed=1
num_workers=8
print_every=100

warmup_steps=-1
warmup_epochs=1

# ============================================================================
# 【3】早停参数
# ============================================================================
es_flag=1
es_min_epochs=10
es_patience=20
es_metric='loss'

# ============================================================================
# 【4】模型架构参数
# ============================================================================
model_tuple='PANTHER,default'

# ========== 原型相关 - 🔴 重要修改! ==========
# 原型模式选择
prototype_mode='leiden'           # 'leiden' 或 'kmeans'
                                  # leiden: 使用Leiden聚类(HPL方法),自动确定原型数
                                  # kmeans: 使用K-means聚类,固定原型数

# K-means参数(仅当prototype_mode='kmeans'时使用)
n_proto_fixed=16                  # 固定原型数量(仅kmeans模式)

# Leiden参数(仅当prototype_mode='leiden'时使用)
leiden_resolution=1.0             # Leiden分辨率(用于生成原型时的值)
proto_num_samples='1.0e+05'       # 原型采样数量

# 原型加载设置
load_proto=1                      # 是否加载预训练原型
fix_proto=1                       # 是否固定原型
out_type='allcat'                 # 输出类型

# EM算法参数
em_step=5
tau=1.0
eps=1

# ============================================================================
# 【5】多模态融合参数
# ============================================================================
model_mm_type='coattn'
num_coattn_layers=1
append_embed='random'
append_prob=0
histo_agg='mean'
net_indiv=1

# ============================================================================
# 【6】基因组学参数
# ============================================================================
omics_modality='pathway'
type_of_path='hallmarks'
omics_dir='data_csvs/rna'

# ============================================================================
# 【7】损失函数参数
# ============================================================================
loss_fn='cox'
n_label_bin=4
alpha=0.5

# ============================================================================
# 【8】数据采样参数
# ============================================================================
bag_size='-1'
train_bag_size='-1'
val_bag_size='-1'

# ============================================================================
# 【9】日志和保存参数
# ============================================================================
save_dir_root=results
wandb_project='mmp_final'
overwrite=0
tags=''

# ============================================================================
# 【10】实验标识参数
# ============================================================================
exp_code=''

# ============================================================================
# 脚本自动生成部分
# ============================================================================

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=${feat}

# 识别特征路径
all_feat_dirs=""
for dataroot_path in "${dataroots[@]}"; do
  feat_dir=${dataroot_path}/extracted_mag${mag}_patch${patch_size}_fp/${feat}/feats_h5
  if ! test -d $feat_dir; then
    continue
  fi
  if [[ -z ${all_feat_dirs} ]]; then
    all_feat_dirs=${feat_dir}
  else
    all_feat_dirs=${all_feat_dirs},${feat_dir}
  fi
done

# ========== 🔴 关键修改: 根据原型模式确定原型路径和数量 ==========
if [[ ${prototype_mode} == 'leiden' ]]; then
    # Leiden模式: 使用Leiden生成的原型
    mode_str="leiden_res${leiden_resolution}"
    
    # 构建原型文件路径(需要匹配prototype生成时的命名)
    proto_path="splits/${split_dir}/prototypes/prototypes_c*_${feat_name}_${mode_str}_num_${proto_num_samples}.pkl"
    
    # 查找匹配的原型文件
    proto_file=$(ls ${proto_path} 2>/dev/null | head -n 1)
    
    if [[ -z ${proto_file} ]]; then
        echo "❌ Error: Leiden prototype file not found!"
        echo "   Expected pattern: ${proto_path}"
        echo ""
        echo "Please run prototype generation first:"
        echo "  bash scripts/prototype/DMSS_xxx_prototype.sh ... leiden"
        exit 1
    fi
    
    # 从文件名提取原型数量(例如: prototypes_c23_xxx.pkl → 23)
    n_proto=$(echo ${proto_file} | sed -n 's/.*prototypes_c\([0-9]*\)_.*/\1/p')
    
    echo "============================================"
    echo "🔍 Leiden Prototype Mode"
    echo "============================================"
    echo "  Prototype file: ${proto_file}"
    echo "  Number of prototypes: ${n_proto} (auto-detected)"
    echo "  Resolution: ${leiden_resolution}"
    echo "============================================"
    echo ""
    
else
    # K-means模式: 使用固定数量的原型
    n_proto=${n_proto_fixed}
    proto_path="splits/${split_dir}/prototypes/prototypes_c${n_proto}_${feat_name}_faiss_num_${proto_num_samples}.pkl"
    proto_file=${proto_path}
    
    echo "============================================"
    echo "📌 K-means Prototype Mode"
    echo "============================================"
    echo "  Prototype file: ${proto_file}"
    echo "  Number of prototypes: ${n_proto} (fixed)"
    echo "============================================"
    echo ""
fi

# 验证原型文件存在
if [[ ! -f ${proto_file} ]]; then
    echo "❌ Error: Prototype file not found: ${proto_file}"
    exit 1
fi

# 自动生成实验代码
if [[ -z ${exp_code} ]]; then
    if [[ ${prototype_mode} == 'leiden' ]]; then
        exp_code=${task}::${model_config}_leiden_p${n_proto}::${feat_name}
    else
        exp_code=${task}::${model_config}::${feat_name}
    fi
fi

save_dir=${save_dir_root}/${exp_code}

# Warmup逻辑
th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
  warmup=0
  curr_lr_scheduler='constant'
else
  curr_lr_scheduler=$lr_scheduler
  warmup=${warmup_epochs}
fi

echo "Running with configuration:"
echo "  Model: ${model}"
echo "  Multimodal type: ${model_mm_type}"
echo "  Omics modality: ${omics_modality}"
echo "  Prototype mode: ${prototype_mode}"
echo "  Number of prototypes: ${n_proto}"
echo "  EM iterations: ${em_step}"
echo "  Learning rate: ${lr}"
echo "  Max epochs: ${max_epoch}"
echo "  Save directory: ${save_dir}"
echo ""

# ============================================================================
# 构建训练命令
# ============================================================================

cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_survival \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
"

# 模型参数
cmd="${cmd}--model_histo_type ${model} \\
--model_histo_config ${model}_default \\
--in_dim ${input_dim} \\
"

# 训练参数
cmd="${cmd}--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--accum_steps ${grad_accum} \\
--wd ${wd} \\
--warmup_epochs ${warmup} \\
--max_epochs ${max_epoch} \\
--batch_size ${batch_size} \\
--seed ${seed} \\
--num_workers ${num_workers} \\
--print_every ${print_every} \\
"

# 早停参数
if [[ $es_flag -eq 1 ]]; then
  cmd="${cmd}--early_stopping ${es_flag} \\
--es_min_epochs ${es_min_epochs} \\
--es_patience ${es_patience} \\
--es_metric ${es_metric} \\
"
fi

# 数据参数
cmd="${cmd}--train_bag_size ${bag_size} \\
--val_bag_size ${val_bag_size} \\
"

# ========== 🔴 EM和原型参数(使用动态的n_proto) ==========
cmd="${cmd}--em_iter ${em_step} \\
--tau ${tau} \\
--n_proto ${n_proto} \\
--out_type ${out_type} \\
--ot_eps ${eps} \\
"

# 原型加载
if [[ $load_proto -eq 1 ]]; then
  cmd="${cmd}--load_proto \\
--proto_path ${proto_file} \\
"
fi

if [[ $fix_proto -eq 1 ]]; then
  cmd="${cmd}--fix_proto \\
"
fi

# 损失函数
cmd="${cmd}--loss_fn ${loss_fn} \\
--nll_alpha ${alpha} \\
--n_label_bins ${n_label_bin} \\
"

# 多模态参数
cmd="${cmd}--num_coattn_layers ${num_coattn_layers} \\
--model_mm_type ${model_mm_type} \\
--append_embed ${append_embed} \\
--histo_agg ${histo_agg} \\
"

if [[ $net_indiv -eq 1 ]]; then
  cmd="${cmd}--net_indiv \\
"
fi

# 基因组学参数
if [[ -n ${omics_modality} && ${omics_modality} != "None" ]]; then
  cmd="${cmd}--omics_modality ${omics_modality} \\
--type_of_path ${type_of_path} \\
--omics_dir ${omics_dir} \\
"
fi

# 日志参数
cmd="${cmd}--wandb_project ${wandb_project} \\
"

# 执行命令
echo ""
echo "============================================"
echo "Executing command:"
echo "============================================"
eval "$cmd"