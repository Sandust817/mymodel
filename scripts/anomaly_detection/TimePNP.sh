export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /root/data/anomaly_detection/MSL/ \
  --model_id MSL \
  --model TimePNP \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 40 \
  --warm_up 3 \

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
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 40 \
  --warm_up 3 \


python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /root/data/anomaly_detection/SMAP/ \
  --model_id SMAP \
  --model TimePNP \
  --data SMAP \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 40 \
  --warm_up 3 \



python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path /root/data/anomaly_detection/SMD/ \
  --model_id SMD \
  --model TimePNP \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 40 \
  --warm_up 3 \



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
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 51 \
  --c_out 51 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 200 \
  --patience 20 \
  --warm_up 3 \