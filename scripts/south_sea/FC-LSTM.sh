gpu=0
model_name=FC_LSTM
seq_len=90
node_num=1665  # bohai:112 south_sea:1665
hidden_dim=5
area=south_sea

if [ ! -d "./logs" ]; then
  mkdir ./logs
fi
if [ ! -d "./logs/${area}" ]; then
  mkdir ./logs/${area}
fi
if [ ! -d "./logs/${area}/${model_name}" ]; then
  mkdir ./logs/${area}/${model_name}
fi


for pred_len in  30 90 180 365 ; do   
    python -u run_longExp.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./datasets/SST/ \
    --data_path data_${area}_mv.npy \
    --model_id ${area}_${seq_len}_${pred_len} \
    --model $model_name \
    --data ev_custom \
    --type mv\
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
    --batch_size 128\
    --loss mse \
    --itr 1 >logs/${area}/${model_name}/${seq_len}_${pred_len}__hd${hidden_dim}.log
done