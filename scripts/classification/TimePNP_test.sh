export CUDA_VISIBLE_DEVICES=0

model_name=TimePNP


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/AtrialFibrillation/ \
  --model_id AtrialFibrillation \
  --model $model_name \
  --data UEA \
  --e_layers 1 \
  --dropout 0.2 \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 150 \
  --patience 30 \
  --K 2

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/LSST/ \
  --model_id LSST \
  --model $model_name \
  --data UEA \
  --e_layers 1 \
  --dropout 0.2 \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 150 \
  --patience 30

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 1 \
  --dropout 0.2 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 150 \
  --patience 30

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/StandWalkJump/ \
  --model_id StandWalkJump \
  --model $model_name \
  --data UEA \
  --e_layers 1 \
  --dropout 0.2 \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 150 \
  --patience 30