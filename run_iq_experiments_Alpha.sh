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
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
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


# ##### AGEM ##### [0.66, 0.34] 6H
# python3 -u main.py $IQ --model agem --expt_name all_agem --n_memories 5192 --batch_size 128 --n_epochs 50 \
#                     --lr 0.03 --glances 1 --memory_strength 0.5   --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### ER ##### [0.76, 0.84] 11H
# python3 -u main.py $IQ --model eralg4 --expt_name all_eralg4 --memories 5192 --batch_size 128 --n_epochs 50 --replay_batch_size 64 \
#                      --lr 0.03 --glances 1   --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

##### La-ER ##### [0.78, 0.92] # batch size = 256 is twice as fast as 64 (1800 vs 900 secs/epoch)
# python3 -u main.py $IQ --model eralg4 --expt_name all_la-eralg4 --memories 5192 --batch_size 256 --replay_batch_size 64 --n_epochs 15 \
#                     --opt_lr 0.1 --alpha_init 0.1 --glances 1  --increment 5 \
#                      --cifar_batches 5 --learn_lr --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### iCaRL ##### Problem Child
# python3 -u main.py $IQ --model icarl --expt_name all_icarl --n_memories 5192 --batch_size 128 --n_epochs 50 \
#                     --lr 0.03 --glances 1 --memory_strength 1.0  --increment 5 \
#                     --log_every 3125 --class_order random  --samples_per_task 2500 \
#                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 --samples_per_task -1

#### GEM ##### [0.52, 0.23] Problem Child - NEEDS REDO
python3 -u main.py $IQ --model gem --expt_name all_gem --n_memories 512 --batch_size 128 --n_epochs 50 \
                    --lr 0.03 --glances 1 --memory_strength 0.5   --increment 5 \
                    --log_every 3125 --class_order random --samples_per_task -1 \
                    --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3


# ##### EWC #####
# python3 -u main.py $IQ --model ewc --expt_name all_ewc --batch_size 128 --n_epochs 50 \
#                     --lr 0.03 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### PackNet #####
# python3 -u main.py $IQ --model packnet --expt_name all_packnet --batch_size 128 --n_epochs 50 \
#                     --lr 0.01 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### RWalk #####
# python3 -u main.py $IQ --model rwalk --expt_name all_rwalk --batch_size 128 --n_epochs 50 \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### Synaptic Intelligence #####
# python3 -u main.py $IQ --model si --expt_name all_si --batch_size 128 --n_epochs 50 \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### HAT #####
# python3 -u main.py $IQ --model hat --expt_name all_hat --batch_size 128 --n_epochs 50 \
#                     --lr 0.0005 --gamma 0.75 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --grad_clip_norm 10.0 --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### CTN #####
# python3 -u main.py $IQ --model ctn --expt_name all_ctn --batch_size 128 --replay_batch_size 64 --n_epochs 50 \
#                     --increment 5 \
#                     --ctn_n_memories 5192 --ctn_lr 0.01 --ctn_beta 0.05 --ctn_inner_steps 2 --ctn_n_meta 2 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### ER-Ring #####
# python3 -u main.py $IQ --model er_ring --expt_name all_erring --batch_size 128 --replay_batch_size 64 --n_epochs 50 \
#                     --lr 0.03 --increment 5 \
#                     --bcl_n_memories 5192 --bcl_temperature 2.0 --bcl_memory_strength 1.0 --bcl_inner_steps 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

##### BCL Dual ##### Hyperparams from logs/bcl_dual/training_parameters.json
python3 -u main.py $IQ --model bcl_dual --expt_name bcl_basic_test --data_path data/rff/rfmls \
                    --n_layers 2 --n_hiddens 100 --xav_init --glances 1 --n_epochs 20 \
                    --batch_size 64 --replay_batch_size 10 --memories 400 --lr 0.03 \
                    --increment 5 --log_every 3125 --class_order random \
                    --seed $SEED --validation 0.2 --samples_per_task -1 --classes_per_it 6 \
                    --iterations 5000 --test_batch_size 100000 --n_memories 5192 --memory_strength 1.0 \
                    --steps_per_sample 1 --gamma 1.0 --beta 0.1 --batches_per_example 1 \
                    --opt_lr 0.1 --opt_wt 0.1 --alpha_init 0.1 --cifar_batches 3 --grad_clip_norm 5.0 \
                    --second_order --bcl_n_memories 2000 --bcl_memory_strength 1.0 --bcl_temperature 2.0 \
                    --bcl_inner_steps 5 --bcl_n_meta 5 --bcl_adapt_lr 0.1 --train_mc_iters 2 \
                    --std_init 0.02 --mean_eta 50.0 --fisher_gamma 0.95 --rln 7 --update_steps 10 \
                    --meta_lr 0.001 --update_lr 0.1

# ##### UCL-BResNet #####
# python3 -u main.py $IQ --model ucl_bresnet --expt_name all_ucl_bresnet --batch_size 64 --n_epochs 50 \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### ANML ##### Slow
# python3 -u main.py $IQ --model anml --expt_name all_anml --batch_size 128 --n_epochs 50 \
#                     --lr 0.001 --glances 1 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1 \
#                     --rln 7 --update_steps 10 --meta_lr 0.001 --update_lr 0.1 

# ##### LwF #####
# python3 -u main.py $IQ --model lwf --expt_name all_lwf --batch_size 128 --n_epochs 50 \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

##### MER ##### VEEEEERY SLOW
python3 -u main.py $IQ --model meralg1 --expt_name all_meralg1 --batch_size 128 --memories 5192 --replay_batch_size 64 --n_epochs 15\
                    --lr 0.1 --beta 0.1 --gamma 1.0 --batches_per_example 10  --increment 5 \
                    --log_every 3125 --grad_clip_norm 10.0 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### Meta BGD ##### [0.24, 0.39] 
# python3 -u main.py $IQ --model meta-bgd --expt_name all_meta-bgd --memories 5192 --batch_size 128 --replay_batch_size 64 --n_epochs 15 \
#                     --alpha_init 0.1 --glances 1  --increment 5 \
#                     --cifar_batches 3 --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1 --xav_init  --std_init 0.02 --mean_eta 50. --train_mc_iters 2

# ##### La-MAML ##### [0.41, 0.48]
# python3 -u main.py $IQ --model lamaml_cifar --expt_name all_lamaml --memories 5192 --batch_size 128 --replay_batch_size 64 --n_epochs 15 \
#                     --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 1  --increment 5 \
#                     --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### sync ##### [0.52, 0.52]
# python3 -u main.py $IQ --model lamaml_cifar --expt_name all_lamaml_sync --memories 5192 --batch_size 128 --replay_batch_size 64 --n_epochs 15 \
#                     --opt_lr 0.35 --alpha_init 0.1 --opt_wt 0.1 --glances 1  --increment 5 \
#                     --cifar_batches 5 --learn_lr --sync_update --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1

# ##### C-MAML ##### [0.45, 0.51]
# python3 -u main.py $IQ --model lamaml_cifar --expt_name all_cmaml --memories 5192 --batch_size 128 --replay_batch_size 64 --n_epochs 15 \
#                     --opt_lr 0.35 --alpha_init 0.075 --opt_wt 0.075 --glances 1  --increment 5 \
#                     --cifar_batches 5 --sync_update --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task -1
