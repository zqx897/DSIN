if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi
gpu=1
model_name=PathFormer
seq_len=96
for pred_len in 96 192 336 720; do   # 
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
    --patch_size_list 16 12 8 32 12 8 6 32 \
    --num_nodes 112 \
    --layer_nums 2 \
    --k 2\
    --d_model 16 \
    --d_ff 64 \
    --train_epochs 10\
    --patience 3\
    --lradj 'TST' \
    --des 'Exp' \
    --gpu $gpu \
    --learning_rate 0.001 \
    --itr 1 >logs/LongForecasting/${model_name}_bohai_${seq_len}_${pred_len}.log
done


# crossgnn的enc_in等价于pathformer的num_nodes，都是节点（变量）数量
#--lradj 'TST' 原论文使用TST、crossgnn使用type1