model_name=Transformer
type=sv
area=bohai
if [ ! -d "./logs" ]; then
  mkdir ./logs
fi
if [ ! -d "./logs/${area}" ]; then
  mkdir ./logs/${area}
fi

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_bohai_mv.npy \
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
  --enc_in 112 \
  --dec_in 112 \
  --c_out 112 \
  --des 'Exp' \
  --gpu 0 \
  --itr 1 \
  --train_epochs 30 >logs/${area}/${model_name}_90_30.log

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_bohai_mv.npy \
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
  --enc_in 112 \
  --dec_in 112 \
  --c_out 112 \
  --des 'Exp' \
  --gpu 0 \
  --itr 1 \
  --train_epochs 30 >logs/${area}/${model_name}_90_90.log

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_bohai_mv.npy \
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
  --enc_in 112 \
  --dec_in 112 \
  --c_out 112 \
  --des 'Exp' \
  --gpu 0 \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 >logs/${area}/${model_name}_90_180.log

python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_bohai_mv.npy \
  --model_id ${area}_90_365 \
  --model $model_name \
  --data sv_custom \
  --type $type \
  --features M \
  --seq_len 90 \
  --label_len 45 \
  --pred_len 365 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --lradj 'TST' \
  --enc_in 112 \
  --dec_in 112 \
  --c_out 112 \
  --des 'Exp' \
  --gpu 0 \
  --itr 1 \
  --train_epochs 30 >logs/${area}/${model_name}_90_365.log