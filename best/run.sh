export CUDA_VISIBLE_DEVICES=0

model_name=TimePNP

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path /root/data/UEA/EthanolConcentration/ \
  --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_EthanolConcentration_TimePNP_UEA_ftM_sl1751_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/FaceDetection/ \
  --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_FaceDetection_TimePNP_UEA_ftM_sl62_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/Handwriting/ \
  --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_Handwriting_TimePNP_UEA_ftM_sl152_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/Heartbeat/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_Heartbeat_TimePNP_UEA_ftM_sl405_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/JapaneseVowels/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_JapaneseVowels_TimePNP_UEA_ftM_sl29_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/PEMS-SF/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_PEMS-SF_TimePNP_UEA_ftM_sl144_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/SelfRegulationSCP1/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_SelfRegulationSCP1_TimePNP_UEA_ftM_sl896_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/SelfRegulationSCP2/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_SelfRegulationSCP2_TimePNP_UEA_ftM_sl1152_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/SpokenArabicDigits/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_SpokenArabicDigits_TimePNP_UEA_ftM_sl93_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
  --is_training 0 \
  --root_path /root/data/UEA/UWaveGestureLibrary/ \
    --ckpt_path /root/sxh/mymodel2/mymodel/best/classification_UWaveGestureLibrary_TimePNP_UEA_ftM_sl315_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
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
