if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/Ablation" ]; then
  mkdir ./logs/Ablation
fi

if [ ! -d "./logs/Ablation/south_sea" ]; then
  mkdir ./logs/Ablation/south_sea
fi

if [ ! -d "./logs/Ablation/bohai" ]; then
  mkdir ./logs/Ablation/bohai
fi
gpu=6
model_name=snMoE_CrossGNN
seq_len=90
node_num=112 # 112:bohai   1665:south_sea
area=bohai
d_model=24
loss=weighted 
for tk in  5 ; do
  for pred_len in 30 90; do   
      python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/SST/ \
      --data_path data_${area}_mv.npy \
      --model_id ${area}_${seq_len}_${pred_len}_dm${d_model} \
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
      --d_model $d_model\
      --train_epochs 20\
      --patience 5\
      --lradj 'TST' \
      --des 'Exp' \
      --gpu $gpu \
      --learning_rate 0.01 \
      --batch_size 16\
      --loss $loss \
      --itr 1 >logs/Ablation/${area}/${model_name}_${loss}_dm${d_model}_${seq_len}_${pred_len}_tk${tk}.log
  done
done
fi
