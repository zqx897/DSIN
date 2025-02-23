if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/FC_LSTM" ]; then
  mkdir ./logs/FC_LSTM
fi
if [ ! -d "./logs/FC_LSTM/south_sea" ]; then
  mkdir ./logs/FC_LSTM/south_sea
fi
if [ ! -d "./logs/FC_LSTM/bohai" ]; then
  mkdir ./logs/FC_LSTM/bohai
fi
gpu=6
model_name=FC_LSTM
seq_len=90
node_num=112   # bohai:112 south_sea:1665
hidden_dim=5
area=bohai

for pred_len in  30 90 180 365 ; do   
    python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/SST/ \
    --data_path data_${area}_mv.npy \
    --model_id ${area}_${seq_len}_${pred_len} \
    --model $model_name \
    --data ev_custom \
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
    --itr 1 >logs/FC_LSTM/${area}/${model_name}_${seq_len}_${pred_len}_hidden_dim${hidden_dim}.log
done