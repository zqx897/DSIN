gpu=0
area=bohai
model_name=EA_CrossGNN
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
d_model=128
hidden=128
global_hidden=5
loss=mse 
for tk in  5 ; do
  for pred_len in 30 90 180 365; do   
      python -u run_longExp.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./datasets/SST/ \
      --data_path data_${area}_mv.npy \
      --model_id ${area}_${seq_len}_${pred_len}_dm${d_model} \
      --model $model_name \
      --data ev_custom \
      --type mv \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --tk $tk\
      --num_nodes $node_num \
      --enc_in $node_num \
      --layer_nums 2 \
      --scale_number 4\
      --k 4\
      --hidden $hidden\
      --d_model $d_model\
      --train_epochs 30\
      --patience 3\
      --lradj 'TST' \
      --des 'Exp' \
      --gpu $gpu \
      --learning_rate 0.0001 \
      --batch_size 32\
      --loss $loss \
      --global_hidden $global_hidden\
      --itr 1 >logs/${area}/${model_name}/${seq_len}_${pred_len}_${loss}_dm${d_model}_tk${tk}_hd${hidden}_gh${global_hidden}.log
  done
done

