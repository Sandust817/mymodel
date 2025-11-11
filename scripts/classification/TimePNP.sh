export CUDA_VISIBLE_DEVICES=0

model_name=TimePNP

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model TimePNP \
  --data UEA \
  --e_layers 3 \
  --batch_size 64 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 32 \
  --dropout 0.2 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --patience 20
