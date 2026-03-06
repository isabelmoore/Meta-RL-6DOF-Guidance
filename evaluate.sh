#!/bin/bash

RUN_DIR="training_logs/Mar03_0115_A_int-gaudet_tar-f16_pro_nav_rew-gaudet_30m"
SCENARIOS="A"
EPISODES=5
GIFS=2
GPU=0

# Build eval dir name (matches evaluate_meta.py naming)
RUN_NAME=$(basename "$RUN_DIR")
SCEN_TAG=$(echo $SCENARIOS | tr ' ' '_')
GIFS_TAG=""
DATE_TAG=$(date +"%b%d_%H%M") # datetime.now().strftime("%b%d_%H%M")
if [ "$GIFS" -gt 0 ]; then GIFS_TAG="_${GIFS}gifs"; fi
EVAL_DIR="evaluate_logs/${DATE_TAG}_${RUN_NAME}_eval_${SCEN_TAG}_${EPISODES}ep${GIFS_TAG}"
mkdir -p "$EVAL_DIR"
LOG_FILE="$EVAL_DIR/evaluate.log"

echo "Evaluating: $RUN_NAME scenarios=$SCENARIOS episodes=$EPISODES gifs=$GIFS"
echo "Eval dir: $EVAL_DIR"
echo "Log: $LOG_FILE"

nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU python3 evaluate_meta.py \
    --run '$RUN_DIR' \
    --scenarios $SCENARIOS \
    --episodes $EPISODES \
    --gifs $GIFS \
    --eval-dir '$EVAL_DIR'" > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "tail -f $LOG_FILE"
