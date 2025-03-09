gpu=2
model_name=ConvLSTM
seq_len=90
node_num=112  # bohai:112 south_sea:1665
hidden_dim=64
area=bohai

if [ ! -d "./logs" ]; then
  mkdir ./logs
fi
if [ ! -d "./logs/${area}" ]; then
  mkdir ./logs/${area}
fi
if [ ! -d "./logs/${area}/${model_name}" ]; then
  mkdir ./logs/${area}/${model_name}
fi

for pred_len in 30 90 180 365; do   #  
    python -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./datasets/SST/ \
    --data_path data_${area}_grid_interpolated_sst.npy\
    --model_id ${area}_${seq_len}_${pred_len} \
    --model $model_name \
    --data grid_Custom \
    --features M \
    --hidden_dim $hidden_dim\
    --seq_len $seq_len \
    --pred_len $pred_len \
    --num_nodes $node_num \
    --train_epochs 30\
    --patience 3\
    --lradj 'TST' \
    --des 'Exp' \
    --gpu $gpu \
    --learning_rate 0.0001 \
    --batch_size 32\
    --loss mse \
    --type grid\
    --itr 1 >logs/${area}/${model_name}/${seq_len}_${pred_len}_hd${hidden_dim}_ks3_nl2.log
done




# crossgnn的enc_in等价于pathformer的num_nodes，都是节点（变量）数量
#--lradj 'TST' 原论文使用TST、crossgnn使用type1
