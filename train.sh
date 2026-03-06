#!/bin/bash

SCENARIOS="A B C" # "A" or "A B" or "A B C"
N_ENVS=8
TIMESTEPS=30000000
GPU=1

# Build run dir name (matches train_meta.py naming)
SCEN_TAG=$(echo $SCENARIOS | tr ' ' '_')
TIMESTEPS_M=$((TIMESTEPS / 1000000))
TIMESTAMP=$(date +"%b%d_%H%M")
RUN_DIR="training_logs/${TIMESTAMP}_META_${SCEN_TAG}_${TIMESTEPS_M}M"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/train.log"

echo "Training: scenarios=$SCENARIOS timesteps=$TIMESTEPS"
echo "Run dir: $RUN_DIR"
echo "Log: $LOG_FILE"

nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU python3 train_meta.py \
    --scenarios $SCENARIOS \
    --n-envs $N_ENVS \
    --timesteps $TIMESTEPS \
    --run-dir $RUN_DIR" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "tail -f $LOG_FILE"
