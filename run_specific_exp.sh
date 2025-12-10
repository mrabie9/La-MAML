#!/bin/bash

# ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_rotations    --cuda --log_dir logs/"
# PERM="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 1000 --dataset mnist_permutations --cuda --log_dir logs/"
# MANY="--n_layers 2 --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_manypermutations --cuda --log_dir logs/"
# CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'
# IMGNET='--data_path data/tiny-imagenet-200/ --log_every 100 --dataset tinyimagenet --cuda --log_dir logs/'

STATE_LOGGING=${STATE_LOGGING:-1}
STATE_LOG_FLAG=""
if [ "$STATE_LOGGING" != "0" ]; then
    STATE_LOG_FLAG="--state_logging"
fi
IQ="--data_path data/rff/radar/ --log_every 100 --dataset iq --cuda --log_dir logs/ --arch resnet1d  --loader task_incremental_loader ${STATE_LOG_FLAG}"
SEED=0

echo "n_memories changed to 1024"
python3 -u main.py $IQ --model gem --expt_name all_gem --n_memories 1024 --batch_size 128 --n_epochs 1 \
                    --lr 0.03 --glances 1 --memory_strength 0.5   --increment 5 \
                    --log_every 3125 --class_order random --samples_per_task -1 \
                    --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 

echo "n_memories changed to 2048"
python3 -u main.py $IQ --model gem --expt_name all_gem --n_memories 2048 --batch_size 128 --n_epochs 1 \
                    --lr 0.03 --glances 1 --memory_strength 0.5   --increment 5 \
                    --log_every 3125 --class_order random --samples_per_task -1 \
                    --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 

echo "n_memories changed to 5096"
python3 -u main.py $IQ --model gem --expt_name all_gem --n_memories 5096 --batch_size 128 --n_epochs 1 \
                    --lr 0.03 --glances 1 --memory_strength 0.5   --increment 5 \
                    --log_every 3125 --class_order random --samples_per_task -1 \
                    --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 