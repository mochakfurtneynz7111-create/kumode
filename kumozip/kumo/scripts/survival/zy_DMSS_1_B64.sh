#!/bin/bash
# ============================================================================
# DMSS Complete Configuration Script
# 包含所有可配置的参数（从main_survival.py提取）
# ============================================================================

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
dataroots=("$@")

# ============================================================================
# 【1】数据和特征提取参数
# ============================================================================
feat='extracted-vit_large_patch16_224.dinov2.uni_mass100k'
input_dim=1536                    # UNI特征维度 (ResNet50: 1024, UNI: 1536)
mag='20x'                         # 放大倍数
patch_size=256                    # Patch大小

# ============================================================================
# 【2】训练基础参数
# ============================================================================
max_epoch=200                     # 最大训练轮数 (default: 20)
lr=0.0001                         # 学习率 (default: 1e-4)
wd=0.00001                        # 权重衰减 (default: 1e-5)
lr_scheduler='cosine'             # 学习率调度器: cosine/linear/constant (default: constant)
opt='adamW'                       # 优化器: adamW/sgd/RAdam (default: adamW)
grad_accum=1                      # 梯度累积步数 (default: 1)
batch_size=64                     # 批次大小 (default: 1)
seed=1                            # 随机种子 (default: 1)
num_workers=8                     # 数据加载线程数 (default: 2)
print_every=100                   # 打印频率 (default: 100)

# Warmup设置
warmup_steps=-1                   # Warmup步数 (default: -1, 不使用)
warmup_epochs=1                   # Warmup轮数 (default: -1, 不使用)

# ============================================================================
# 【3】早停 (Early Stopping) 参数
# ============================================================================
es_flag=1                         # 是否启用早停: 1=启用, 0=禁用 (default: 0)
es_min_epochs=10                  # 早停最小轮数 (default: 10)
es_patience=20                    # 早停耐心值 (default: 20)
es_metric='loss'                  # 早停监控指标: loss/cindex (default: loss)

# ============================================================================
# 【4】模型架构参数
# ============================================================================
# 病理模型
model_tuple='PANTHER,default'     # 模型类型和配置
                                  # 可选: PANTHER/H2T/OT/ProtoCount/MIL
n_fc_layer=0                      # 全连接层数 (default: None)

# 原型相关
out_size=16                       # 原型数量 (n_proto) (default: None)
out_type='allcat'                 # 输出类型: allcat/param_cat (default: param_cat)
load_proto=1                      # 是否加载预训练原型: 1=是, 0=否 (default: False)
fix_proto=1                       # 是否固定原型: 1=是, 0=否 (default: False)
proto_num_samples='1.0e+05'       # 原型采样数量

# EM算法参数
em_step=1                        # EM迭代次数 (default: None)
                                  # DMSS: 1, MMP: 0
tau=1.0                           # 温度参数 (default: None)
                                  # 论文值: 1.0
eps=1                             # OT epsilon (default: 0.1)
                                  # 论文值: 1.0

# ============================================================================
# 【5】多模态融合参数
# ============================================================================
model_mm_type='coattn'            # 多模态融合类型 (default: coattn)
                                  # 可选: coattn/coattn_mot/survpath/histo/gene
                                  # coattn: Co-attention (DMSS, MMP)
                                  # coattn_mot: Co-attention + OT (MOTCat)
                                  # survpath: SurvPath方法
                                  # histo: 只用病理 (单模态)
                                  # gene: 只用基因 (单模态)

num_coattn_layers=1               # Co-attention层数 (default: 1)
append_embed='random'             # 嵌入追加方式 (default: none)
                                  # 可选: none/modality/proto/mp/random
append_prob=0                     # 是否追加概率 (default: False)
histo_agg='mean'                  # 病理特征聚合方式 (default: mean)
net_indiv=1                       # 是否使用独立网络 (default: False)

# ============================================================================
# 【6】基因组学参数 - 🔴 重要！
# ============================================================================
omics_modality='pathway'          # 基因数据模式 (default: pathway)
                                  # 可选: pathway/functional/None
                                  # pathway: 通路数据 (DMSS使用)
                                  # functional: 6个功能组 (MCAT使用)
                                  # None: 不使用基因数据 (单模态)

type_of_path='hallmarks'          # 通路类型 (default: hallmarks)
                                  # 可选: hallmarks/reactome/combine
                                  # hallmarks: 50个Hallmarks通路
                                  # combine: 331个组合通路

omics_dir='data_csvs/rna'         # 基因数据目录 (default: ./data_csvs/rna)

# ============================================================================
# 【7】损失函数参数
# ============================================================================
loss_fn='cox'                     # 损失函数 (default: nll)
                                  # 可选: cox/nll/sumo/ipcwls/rank
                                  # cox: Cox比例风险模型 (DMSS使用)
                                  # nll: 负对数似然

n_label_bin=4                     # 标签分箱数 (仅nll使用) (default: 4)
alpha=0.5                         # NLL alpha平衡参数 (default: 0)

# ============================================================================
# 【8】数据采样参数
# ============================================================================
bag_size='-1'                     # Bag大小: -1表示使用全部 (default: -1)
train_bag_size='-1'               # 训练集bag大小 (default: -1)
val_bag_size='-1'                 # 验证集bag大小 (default: -1)

# ============================================================================
# 【9】日志和保存参数
# ============================================================================
save_dir_root=results             # 结果保存根目录 (default: ./results)
wandb_project='mmp_final'         # Wandb项目名 (default: mmp_final)
overwrite=0                       # 是否覆盖已有结果 (default: False)
tags=''                           # 实验标签 (default: None)

# ============================================================================
# 【10】实验标识参数
# ============================================================================
exp_code=''                       # 实验代码 (default: None)
                                  # 如果为空，会自动生成

# ============================================================================
# 以下为脚本自动生成的参数，通常不需要修改
# ============================================================================

IFS=',' read -r model config_suffix <<< "${model_tuple}"
model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')

# 自动生成实验代码
if [[ -z ${exp_code} ]]; then
    exp_code=${task}::${model_config}::${feat_name}
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

# 识别特征路径
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

echo "Feature directory: $feat_dir"
echo "Running with configuration:"
echo "  Model: ${model}"
echo "  Multimodal type: ${model_mm_type}"
echo "  Omics modality: ${omics_modality}"
echo "  EM iterations: ${em_step}"
echo "  Learning rate: ${lr}"
echo "  Max epochs: ${max_epoch}"

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
--n_fc_layers ${n_fc_layer} \\
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

# EM和原型参数
cmd="${cmd}--em_iter ${em_step} \\
--tau ${tau} \\
--n_proto ${out_size} \\
--out_type ${out_type} \\
--ot_eps ${eps} \\
"

# 原型加载
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

# 原型路径
if [[ $load_proto -eq 1 ]]; then
  proto_path="splits/${split_dir}/prototypes/prototypes_c${out_size}_extracted-${feat_name}_faiss_num_${proto_num_samples}.pkl"
  cmd="${cmd}--load_proto \\
--proto_path ${proto_path} \\
"
fi

# 执行命令
echo ""
echo "============================================"
echo "Executing command:"
echo "============================================"
eval "$cmd"