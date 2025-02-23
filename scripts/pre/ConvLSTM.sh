if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/ConvLSTM" ]; then
  mkdir ./logs/ConvLSTM
fi
if [ ! -d "./logs/ConvLSTM/south_sea" ]; then
  mkdir ./logs/ConvLSTM/south_sea
fi
if [ ! -d "./logs/ConvLSTM/bohai" ]; then
  mkdir ./logs/ConvLSTM/bohai
fi
gpu=6
model_name=ConvLSTM
seq_len=90
node_num=112  # bohai:112 south_sea:1665
hidden_dim=64
area=bohai
for pred_len in 30 90 180 365; do   #  
    python -u run_longExp.py \
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
    --train_epochs 20\
    --patience 3\
    --lradj 'TST' \
    --des 'Exp' \
    --gpu $gpu \
    --learning_rate 0.01 \
    --batch_size 32\
    --loss mse \
    --type grid\
    --itr 1 >logs/ConvLSTM/${area}/${model_name}_${seq_len}_${pred_len}_hidden_dim${hidden_dim}_ks3_nl2.log
done




# crossgnn的enc_in等价于pathformer的num_nodes，都是节点（变量）数量
#--lradj 'TST' 原论文使用TST、crossgnn使用type1
