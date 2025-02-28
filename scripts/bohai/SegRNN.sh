model_name=SegRNN
type=sv
area=bohai
gpu=0
node_num=112
d_model=128
seg_len=45
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
for pred_len in 30 90 180 365
do
python -u run_longExp.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./datasets/SST/ \
  --data_path data_${area}_mv.npy \
  --model_id ${area}_90_30 \
  --model $model_name \
  --data sv_custom \
  --type $type \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len $seg_len \
  --enc_in $node_num \
  --d_model $d_model \
  --dropout 0 \
  --learning_rate 0.001 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --train_epochs 1 >logs/${area}/${model_name}/${seq_len}_${pred_len}_dm${d_model}_sl${seg_len}.log
done

