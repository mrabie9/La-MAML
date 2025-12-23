#!/bin/bash

# Quick IQ experiment sweep with n_epochs=1 for smoke testing.

STATE_LOGGING=${STATE_LOGGING:-1}
STATE_LOG_FLAG=""
if [ "$STATE_LOGGING" != "0" ]; then
    STATE_LOG_FLAG="--state_logging"
fi

IQ="--data_path data/rff/radar/ --log_every 100 --dataset iq --cuda --log_dir logs/ --arch resnet1d --loader task_incremental_loader ${STATE_LOG_FLAG}"
SEED=0
N_EPOCHS=1

LOG_DIR="logs/iq_experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/fast_run_${TIMESTAMP}.log"
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

echo "Logging quick IQ experiment suite to $LOG_FILE"

# # ##### AGEM #####
# python3 -u main.py $IQ --model agem --expt_name all_agem --n_memories 5192 --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.03 --glances 1 --memory_strength 0.5 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### ER #####
# python3 -u main.py $IQ --model eralg4 --expt_name all_eralg4 --memories 5192 --batch_size 128 --n_epochs $N_EPOCHS --replay_batch_size 64 \
#                     --lr 0.03 --glances 1 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### iCaRL #####
# python3 -u main.py $IQ --model icarl --expt_name all_icarl --n_memories 5192 --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.03 --glances 1 --memory_strength 1.0 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 --samples_per_task 256

# ##### GEM #####
# python3 -u main.py $IQ --model gem --expt_name all_gem --n_memories 512 --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.03 --glances 1 --memory_strength 0.5 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --grad_clip_norm 5.0 --calc_test_accuracy --validation 0.3 --samples_per_task 256

# ##### EWC #####
# python3 -u main.py $IQ --model ewc --expt_name all_ewc --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.03 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### LwF #####
# python3 -u main.py $IQ --model lwf --expt_name all_lwf --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### PackNet #####
# python3 -u main.py $IQ --model packnet --expt_name all_packnet --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.01 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### RWalk #####
# python3 -u main.py $IQ --model rwalk --expt_name all_rwalk --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### Synaptic Intelligence #####
# python3 -u main.py $IQ --model si --expt_name all_si --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### HAT #####
# python3 -u main.py $IQ --model hat --expt_name all_hat --batch_size 128 --n_epochs $N_EPOCHS \
#                     --lr 0.0005 --gamma 0.75 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --grad_clip_norm 10.0 --calc_test_accuracy --validation 0.3 --samples_per_task 256

# ##### CTN #####
# python3 -u main.py $IQ --model ctn --expt_name all_ctn --batch_size 128 --replay_batch_size 64 --n_epochs $N_EPOCHS \
#                     --increment 5 \
#                     --ctn_n_memories 5192 --ctn_lr 0.01 --ctn_beta 0.05 --ctn_inner_steps 2 --ctn_n_meta 2 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

##### ER-Ring #####
# python3 -u main.py $IQ --model er_ring --expt_name all_erring --batch_size 128 --replay_batch_size 64 --n_epochs $N_EPOCHS \
#                     --lr 0.03 --increment 5 \
#                     --bcl_n_memories 5192 --bcl_temperature 2.0 --bcl_memory_strength 1.0 --bcl_inner_steps 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### UCL-BResNet #####
# python3 -u main.py $IQ --model ucl_bresnet --expt_name all_ucl_bresnet --batch_size 64 --n_epochs $N_EPOCHS \
#                     --lr 0.001 --increment 5 \
#                     --log_every 3125 --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### La-MAML #####
# python3 -u main.py $IQ --model lamaml_cifar --expt_name all_lamaml --memories 5192 --batch_size 128 --replay_batch_size 64 --n_epochs $N_EPOCHS \
#                     --opt_lr 0.25 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --increment 5 \
#                     --cifar_batches 5 --learn_lr --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### sync #####
# python3 -u main.py $IQ --model lamaml_cifar --expt_name all_lamaml_sync --memories 1024 --batch_size 128 --replay_batch_size 64 --n_epochs $N_EPOCHS \
#                     --opt_lr 0.35 --alpha_init 0.1 --opt_wt 0.1 --glances 1 --increment 5 \
#                     --cifar_batches 5 --learn_lr --sync_update --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### C-MAML #####
# python3 -u main.py $IQ --model lamaml_cifar --expt_name all_cmaml --memories 1024 --batch_size 128 --replay_batch_size 64 --n_epochs $N_EPOCHS \
#                     --opt_lr 0.35 --alpha_init 0.075 --opt_wt 0.075 --glances 1 --increment 5 \
#                     --cifar_batches 5 --sync_update --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# ##### MER #####
python3 -u main.py $IQ --model meralg1 --expt_name all_meralg1 --batch_size 128 --memories 5192 --replay_batch_size 64 --n_epochs $N_EPOCHS \
                    --lr 0.1 --beta 0.1 --gamma 1.0 --batches_per_example 10 --increment 5 \
                    --log_every 3125 --grad_clip_norm 10.0 --class_order random \
                    --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256

# # ##### Meta BGD #####
# python3 -u main.py $IQ --model meta-bgd --expt_name all_meta-bgd --memories 5192 --batch_size 128 --replay_batch_size 64 --n_epochs $N_EPOCHS \
#                     --alpha_init 0.1 --glances 1 --increment 5 \
#                     --cifar_batches 3 --log_every 3125 --second_order --class_order random \
#                     --seed $SEED --calc_test_accuracy --validation 0.3 --samples_per_task 256 --xav_init --std_init 0.02 --mean_eta 50. --train_mc_iters 2

# ##### BCL-Dual #####
# python3 -u main.py $IQ --model bcl_dual --expt_name bcl_basic_test --data_path data/rff/rfmls \
#                     --n_layers 2 --n_hiddens 100 --xav_init --glances 1 --n_epochs 1 \
#                     --batch_size 64 --replay_batch_size 10 --memories 400 --lr 0.03 \
#                     --increment 5 --log_every 3125 --class_order random \
#                     --seed $SEED --validation 0.2 --samples_per_task 256 --classes_per_it 6 \
#                     --iterations 5000 --test_batch_size 100000 --n_memories 5192 --memory_strength 1.0 \
#                     --steps_per_sample 1 --gamma 1.0 --beta 0.1 --batches_per_example 1 \
#                     --opt_lr 0.1 --opt_wt 0.1 --alpha_init 0.1 --cifar_batches 3 --grad_clip_norm 5.0 \
#                     --second_order --bcl_n_memories 2000 --bcl_memory_strength 1.0 --bcl_temperature 2.0 \
#                     --bcl_inner_steps 5 --bcl_n_meta 5 --bcl_adapt_lr 0.1 --train_mc_iters 2 \
#                     --std_init 0.02 --mean_eta 50.0 --fisher_gamma 0.95 --rln 7 --update_steps 10 \
#                     --meta_lr 0.001 --update_lr 0.1
