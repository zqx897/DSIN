gpu=3
area=bohai
model_name=CrossGNN
if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/${area}" ]; then
  mkdir ./logs/${area}
fi

if [ ! -d "./logs/${area}/${model_name}" ]; then
  mkdir ./logs/${area}/${model_name}
fi

seq_len=90
node_num=112 # 112:bohai   1665:south_sea
hidden=64
for tk in  5 ; do
  for pred_len in 30 90 180 365; do   
      python -u run_longExp.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./datasets/SST/ \
      --data_path data_${area}_mv.npy \
      --model_id ${area}_${seq_len}_${pred_len}_dm${hidden} \
      --model $model_name \
      --data sv_custom \
      --type sv \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --tk $tk\
      --num_nodes $node_num \
      --enc_in $node_num \
      --layer_nums 2 \
      --scale_number 4\
      --hidden $hidden\
      --train_epochs 30\
      --patience 3\
      --lradj 'TST' \
      --des 'Exp' \
      --gpu $gpu \
      --learning_rate 0.0001 \
      --batch_size 16\
      --itr 1 >logs/${area}/${model_name}/${seq_len}_${pred_len}_hd${hidden}_tk${tk}.log
  done
done

