export CUDA_VISIBLE_DEVICES=0

model_name=TimePNP

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/ArticularyWordRecognition/ \
  --model_id ArticularyWordRecognition \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/AtrialFibrillation/ \
  --model_id AtrialFibrillation \
  --model $model_name \
  --data UEA \
  --e_layers 1 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 128 \
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
  --root_path /root/data/UEA/BasicMotions/ \
  --model_id BasicMotions \
  --model TimePNP \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path /root/data/UEA/CharacterTrajectories/ \
#   --model_id CharacterTrajectories \
#   --model $model_name \
#   --data UEA \
#   --e_layers 2 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0005 \
#   --train_epochs 150 \
#   --patience 30

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Cricket/ \
  --model_id Cricket \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/DuckDuckGeese/ \
  --model_id DuckDuckGeese \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/EigenWorms/ \
  --model_id EigenWorms \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Epilepsy/ \
  --model_id Epilepsy \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/ERing/ \
  --model_id ERing \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/FingerMovements/ \
  --model_id FingerMovements \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/HandMovementDirection/ \
  --model_id HandMovementDirection \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.003 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 256 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/InsectWingbeat/ \
  --model_id InsectWingbeat \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
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
  --root_path /root/data/UEA/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/Libras/ \
  --model_id Libras \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/LSST/ \
  --model_id LSST \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/MotorImagery/ \
  --model_id MotorImagery \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/NATOPS/ \
  --model_id NATOPS \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/PenDigits/ \
  --model_id PenDigits \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/PhonemeSpectra/ \
  --model_id PhonemeSpectra \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 64 \
  --d_model 128 \
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
  --root_path /root/data/UEA/RacketSports/ \
  --model_id RacketSports \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/StandWalkJump/ \
  --model_id StandWalkJump \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path /root/data/UEA/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0005 \
  --train_epochs 150 \
  --patience 30

