model_name=Transformer
type=sv
area=south_sea
gpu=0
node_num=1665
d_model=64
if [ ! -d "./logs" ]; then
  mkdir ./logs
fi
if [ ! -d "./logs/${area}" ]; then
  mkdir ./logs/${area}
fi
if [ ! -d "./logs/${area}/${model_name}" ]; then
  mkdir ./logs/${area}/${model_name}
fi

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_${area}_mv.npy \
  --model_id ${area}_90_30 \
  --model $model_name \
  --data sv_custom \
  --type $type \
  --features M \
  --seq_len 90 \
  --label_len 45 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --lradj 'TST' \
  --enc_in $node_num \
  --dec_in $node_num \
  --c_out $node_num \
  --d_model $d_model \
  --des 'Exp' \
  --gpu $gpu \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 >logs/${area}/${model_name}/90_30_dm${d_model}.log

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_${area}_mv.npy \
  --model_id ${area}_90_90 \
  --model $model_name \
  --data sv_custom \
  --type $type \
  --features M \
  --seq_len 90 \
  --label_len 45 \
  --pred_len 90 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --lradj 'TST' \
  --enc_in $node_num \
  --dec_in $node_num \
  --c_out $node_num \
  --d_model $d_model \
  --des 'Exp' \
  --gpu $gpu \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 >logs/${area}/${model_name}/90_90_dm${d_model}.log

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_${area}_mv.npy \
  --model_id ${area}_90_180 \
  --model $model_name \
  --data sv_custom \
  --type $type \
  --features M \
  --seq_len 90 \
  --label_len 45 \
  --pred_len 180 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --lradj 'TST' \
  --enc_in $node_num \
  --dec_in $node_num \
  --c_out $node_num \
  --d_model $d_model \
  --des 'Exp' \
  --gpu $gpu \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 >logs/${area}/${model_name}/90_180_dm${d_model}.log

# python -u run_longExp.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./datasets/SST/ \
#   --data_path data_${area}_mv.npy \
#   --model_id ${area}_90_365 \
#   --model $model_name \
#   --data sv_custom \
#   --type $type \
#   --features M \
#   --seq_len 90 \
#   --label_len 45 \
#   --pred_len 365 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --lradj 'TST' \
#   --enc_in $node_num \
#   --dec_in $node_num \
#   --c_out $node_num \
#   --d_model $d_model \
#   --des 'Exp' \
#   --gpu $gpu \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --train_epochs 30 >logs/${area}/${model_name}/90_365_dm${d_model}.log