# 指定使用第0块GPU（可根据需求修改为0,1 或 1等）
export CUDA_VISIBLE_DEVICES=0

# 核心参数定义
model_name="TimePNP"
# 可选：若需要统一实验ID，可在这里定义（与原有脚本保持一致性）
# exp_id="UCR_TimeMIL_Exp"

# 2. 批量执行训练（UCR数据集列表，包含经典UCR时间序列分类数据集）
# 注：该列表包含UCR常用基准数据集，可根据你的本地数据集存放情况增删
datasets=(
    "AllGestureWiimoteX" "AllGestureWiimoteZ"
     "Crop" "EOGVerticalSignal"
    "Haptics" "InlineSkate" "MiddlePhalanxTW"
     "PLAID" "PhalangesOutlinesCorrect" "Phoneme"
    "PigAirwayPressure" "PigArtPressure" "SonyAIBORobotSurface2" "Yoga"

)

for dataset in "${datasets[@]}"; do
    python -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "/root/data/UCR/${dataset}/" \
      --model_id "${dataset}" \
      --model "${model_name}" \
      --data UCR \
      --e_layers 2 \
      --batch_size 32 \
      --dropout 0.2 \
      --d_model 256 \
      --d_ff 256 \
      --top_k 3 \
      --des "Exp" \
      --itr 1 \
      --learning_rate 0.001 \
      --train_epochs 150 \
      --patience 30
done