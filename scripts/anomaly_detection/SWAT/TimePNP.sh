export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /root/data/anomaly_detection/SWaT/ \
  --model_id SWAT \
  --model TimePNP \
  --data SWAT \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 40 \
  --warm_up 2 \