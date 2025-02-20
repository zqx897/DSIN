if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/mse_EA_MoE_CrossGNN" ]; then
  mkdir ./logs/mse_EA_MoE_CrossGNN
fi
gpu=2
model_name=EA_snMoE_CrossGNN
seq_len=90
node_num=112
for tk in  5 ; do
  for pred_len in 30 90 180 365 ; do   # 
      python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/SST/ \
      --data_path bohai_mv.npy \
      --model_id bohai_${seq_len}_${pred_len} \
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
      --d_model 16 \
      --train_epochs 10\
      --patience 3\
      --lradj 'TST' \
      --des 'Exp' \
      --gpu $gpu \
      --learning_rate 0.01 \
      --batch_size 32 \
      --loss mse\
      --itr 1 >logs/mse_EA_MoE_CrossGNN/revin0_${model_name}_bohai_${seq_len}_${pred_len}_tk${tk}_k4_3.log
  done
done



# crossgnn的enc_in等价于pathformer的num_nodes，都是节点（变量）数量
#--lradj 'TST' 原论文使用TST、crossgnn使用type1
