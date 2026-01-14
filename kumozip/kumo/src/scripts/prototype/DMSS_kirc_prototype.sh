#!/bin/bash

gpuid=$1

declare -a dataroots=(
	'/root/autodl-tmp/MMP/ALL_embeddings/ALL_KIRC_embeddings'
)

# Loop through different folds
for k in 0 1 2 3 4; do
	split_dir="survival/TCGA_KIRC_overall_survival_k=${k}"
	split_names="train"
	bash "./scripts/prototype/DMSS_clustering.sh" $gpuid $split_dir $split_names "${dataroots[@]}"
done