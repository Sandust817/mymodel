#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 核心参数定义
model_name="TimeMIL"


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
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 100 \
      --patience 20
done
