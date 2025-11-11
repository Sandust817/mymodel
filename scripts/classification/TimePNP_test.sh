export CUDA_VISIBLE_DEVICES=0

model_name=TimePNP


# python -u run.py \
#   --task_name classification \
#   --is_training 0 \
#   --ckpt_path "/root/sxh/mymodel2/Time-Series-Library/checkpoints/classification_FaceDetection_TimePNP_UEA_ftM_sl62_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth" \
#   --root_path /root/data/UEA/FaceDetection/ \
#   --model_id FaceDetection \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --train_epochs 200 \
#   --patience 20


python -u run.py \
  --task_name classification \
  --is_training 0 \
  --ckpt_path "/root/sxh/mymodel2/Time-Series-Library/checkpoints/classification_EthanolConcentration_TimePNP_UEA_ftM_sl1751_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth" \
  --root_path /root/data/UEA/EthanolConcentration/ \
  --model_id EthanolConcentration \
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

