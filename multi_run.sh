#!/bin/bash

lr1=$1
bs1=$2
lr2=$3
bs2=$4
lr3=$5
bs3=$6
train_size1=$7
train_size2=$8
train_size3=$9


CUDA_VISIBLE_DEVICES=0 python -u ft_script_lorec.py --train_size $train_size1 --test_range 0:30000 --K 10 --ctr_K 60 --train_type sequential --test_type sequential --dataset GoodReads --mode origin --model_name vicuna-7b --layers [q,v] --lr $lr1 --bs $bs1 --weight_decay 0  --lora_dropout 0 --head lm --enable_softmax --lora_assemble_num 8 --temperature 1 --epochs 5

if [ $? -ne 0]; then
  echo "ft_script_lorec.py failed"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -u ft_script_lorec.py --train_size $train_size2 --test_range 0:30000 --K 10 --ctr_K 60 --train_type sequential --test_type sequential --dataset GoodReads --mode origin --model_name vicuna-7b --layers [q,v] --lr $lr2 --bs $bs2 --weight_decay 0  --lora_dropout 0 --head lm --enable_softmax --lora_assemble_num 8 --temperature 1 --epochs 5

if [ $? -ne 0]; then
  echo "ft_script_lorec.py failed"
  exit 1
fi


# CUDA_VISIBLE_DEVICES=0 python -u ft_script_lorec.py --train_size $train_size3 --test_range 0:30000 --K 10 --ctr_K 60 --train_type sequential --test_type sequential --dataset GoodReads --mode ctr --model_name vicuna-7b --layers [q,v] --lr $lr3 --bs $bs3 --weight_decay 0  --lora_dropout 0 --head lm --enable_softmax --lora_assemble_num 8 --temperature 1 --epochs 5

# if [ $? -ne 0]; then
#   echo "ft_script_lorec.py failed"
#   exit 1
# fi

# CUDA_VISIBLE_DEVICES=0 python -u ft_script_lorec.py --train_size $train_size3 --test_range 0:1772 --K 10 --ctr_K 60 --train_type sequential --test_type sequential --dataset BookCrossing --mode origin --model_name vicuna-7b --layers [q,v] --lr 1e-3 --bs 256 --weight_decay 0  --lora_dropout 0 --head lm --enable_softmax --lora_assemble_num 8 --temperature 10 --epochs 10 

# if [ $? -ne 0]; then
#   echo "ft_script_lorec.py failed"
#   exit 1
# fi

# CUDA_VISIBLE_DEVICES=0 python -u ft_script_lorec.py --train_size $train_size4 --test_range 0:1772 --K 10 --ctr_K 60 --train_type sequential --test_type sequential --dataset BookCrossing --mode ctr --model_name vicuna-7b --layers [q,v] --lr 1e-3 --bs 256 --weight_decay 0  --lora_dropout 0 --head lm --enable_softmax --lora_assemble_num 16 --temperature 10 --epochs 10 

# if [ $? -ne 0]; then
#   echo "ft_script_lorec.py failed"
#   exit 1
# fi
