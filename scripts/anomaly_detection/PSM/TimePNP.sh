export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /root/data/anomaly_detection/PSM/ \
  --model_id PSM \
  --model TimePNP \
  --data PSM \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 40 \
  --warm_up 2 \