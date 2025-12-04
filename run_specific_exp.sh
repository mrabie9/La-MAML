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

LOG_DIR="logs/iq_experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/problem_children_run_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

awk '
/^[[:space:]]*python3 -u[[:space:]]+main\.py/ {
    cmd=$0
    while (sub(/\\[[:space:]]*$/, "", cmd)) {
        if (getline nextline) {
            cmd = cmd "\n" nextline
        } else {
            break
        }
    }
    print cmd "\n"
}
' "$0"

echo "Logging IQ experiment suite to $LOG_FILE"

####### PROBLEM CHILDREN 3 DEC 25 #######

# ##### ANML ##### Problem Child
# python3 -u main.py $IQ --model anml --expt_name all_anml --batch_size 128 --n_epochs 1 \
#                     --lr 0.001 --glances 1 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 128 \
#                     --rln 7 --update_steps 10 --meta_lr 0.001 --update_lr 0.1 


# ##### iCaRL ##### Problem Child
# python3 -u main.py $IQ --model icarl --expt_name all_icarl --n_memories 200 --batch_size 128 --n_epochs 1 \
#                     --lr 0.03 --glances 1 --memory_strength 1.0  --increment 5 \
#                     --log_every 3125 --class_order random  --samples_per_task 2500 \
#                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 --samples_per_task 128

# #### GEM ##### [0.52, 0.23] Problem Child
# python3 -u main.py $IQ --model gem --expt_name all_gem --n_memories 200 --batch_size 128 --n_epochs 1 \
#                     --lr 0.03 --glances 1 --memory_strength 0.5   --increment 5 \
#                     --log_every 3125 --class_order random --samples_per_task 2500 \
#                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 --samples_per_task 128



#### MER ##### VEEEEERY SLOW
python3 -u main.py $IQ --model meralg1 --expt_name all_meralg1 --batch_size 128 --memories 200 --replay_batch_size 64 \
                    --lr 0.1 --beta 0.1 --gamma 1.0 --batches_per_example 10  --increment 5 \
                    --log_every 3125 --grad_clip_norm 10.0 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 128

# ##### Meta BGD ##### [0.24, 0.39] Problem
# python3 -u main.py $IQ --model meta-bgd --expt_name all_meta-bgd --memories 200 --batch_size 128 --replay_batch_size 64 --n_epochs 1 \
#                     --alpha_init 0.1 --glances 1  --increment 5 \
#                     --cifar_batches 3 --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 128 --xav_init  --std_init 0.02 --mean_eta 50. --train_mc_iters 2