#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 核心参数定义
model_name="TimeMIL"
TZ='Asia/Shanghai'
exp_id=$(date '+%Y%m%d%H')
code_save_root="/root/sxh/mymodel2/Time-Series-Library/checkpoints/_record${exp_id}"
mkdir -p "${code_save_root}"

# 1. 一次性保存代码
code_files=(
    "exp/exp_classification.py exp_classification.py"
    "exp/exp_basic.py exp_basic.py"
    "data_provider/data_factory.py data_factory.py"
    "utils/tools.py tools.py"
    "models/${model_name}.py ${model_name}.py"
    "scripts/classification/${model_name}.sh ${model_name}.sh"
)

for pair in "${code_files[@]}"; do
    src=$(echo "$pair" | awk '{print $1}')  # 提取源路径
    dst=$(echo "$pair" | awk '{print $2}')  # 提取目标文件名
    full_src="/root/sxh/mymodel2/Time-Series-Library/${src}"
    full_dst="${code_save_root}/${dst}"
    if [ -f "$full_src" ]; then
        cp -f "$full_src" "$full_dst"
    fi
done

# 2. 批量执行训练
datasets=(
    "EthanolConcentration" "FaceDetection" "Handwriting" "Heartbeat" "JapaneseVowels"
    "PEMS-SF" "SelfRegulationSCP1" "SelfRegulationSCP2" "SpokenArabicDigits" "UWaveGestureLibrary"
)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UEA/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UEA \
      --e_layers 3 \
      --batch_size 16 \
      --dropout 0.2 \
      --d_model 128 \
      --d_ff 256 \
      --top_k 3 \
      --des "Exp_${exp_id}" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 20
done

echo "All done. Code saved to: ${code_save_root}"