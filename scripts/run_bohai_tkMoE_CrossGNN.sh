if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/tkMOE_CrossGNN" ]; then
  mkdir ./logs/tkMOE_CrossGNN
fi
gpu=0
model_name=tkMoE_CrossGNN
seq_len=96
node_num=112
for scale_number in 1 2 4 ; do
  for pred_len in 96 192 336 720 ; do   # 
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
      --scale_number $scale_number\
      --tk_list  3 5 10 15 3 5 10 15 \
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
      --itr 1 >logs/tkMOE_CrossGNN/revin0_${model_name}_bohai_${seq_len}_${pred_len}_sn${scale_number}_k4.log
  done
done



# crossgnn的enc_in等价于pathformer的num_nodes，都是节点（变量）数量
#--lradj 'TST' 原论文使用TST、crossgnn使用type1
