if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/CMSINet" ]; then
  mkdir ./logs/CMSINet
fi

if [ ! -d "./logs/CMSINet/south_sea" ]; then
  mkdir ./logs/CMSINet/south_sea
fi

gpu=1
model_name=EA_snMoE_CrossGNN
seq_len=90
node_num=1665   # 112:bohai
area=south_sea

for tk in  5 ; do
  for pred_len in 30  ; do   
      python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/SST/ \
      --data_path data_${area}_mv.npy \
      --model_id ${area}_${seq_len}_${pred_len} \
      --model $model_name \
      --data ev_custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --tk $tk\
      --sn_list 1 2 4 8 1 2 4 8 \
      --num_nodes $node_num \
      --enc_in $node_num \
      --layer_nums 2 \
      --k 4\
      --d_model 24 \
      --train_epochs 10\
      --patience 3\
      --lradj 'TST' \
      --des 'Exp' \
      --gpu $gpu \
      --learning_rate 0.01 \
      --batch_size 16\
      --loss weighted \
      --itr 1 >logs/CMSINet/${area}/${model_name}_bohai_${seq_len}_${pred_len}_tk${tk}.log
  done
done
