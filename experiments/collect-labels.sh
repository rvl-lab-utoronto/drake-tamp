#!/bin/bash
set -e
EXPDIR=$1
DOMAIN=$2
PROBLEM_FILE_PATH=$(realpath $3)
PROBLEMS=$(ls $PROBLEM_FILE_PATH/*.yaml)
DIR=$HOME/drake-tamp/
export CUDA_VISIBLE_DEVICES=""

mkdir -p $EXPDIR && cd $EXPDIR && mkdir -p save && mkdir -p oracle
for FILE in $PROBLEMS; do
  echo "Running $FILE"
  RUN=$(basename $FILE)
  LOGDIR=$(realpath ./save/$RUN)_logs/
  timeout --signal 2 --foreground 180s python -O $DIR/experiments/main.py --domain=$DOMAIN --algorithm adaptive --mode normal --logpath $LOGDIR --problem-file $FILE --max_planner_time 10 | tee ./save/$RUN.log
  LOGDIR=$(realpath ./oracle/$RUN)_logs/
  STATSPATH=$(realpath ./save/$RUN)_logs/stats.json
  # STATSPATH: If using mode = oracle after mode =normal, the directory of the stats.json to use for the preimage"
  JSON='{"data_collection_mode":true,"stats_path":"'"$STATSPATH"'"}' # don't add spaces to this or it will break
  timeout --signal 2 --foreground 180s python -O $DIR/experiments/main.py --domain=$DOMAIN --algorithm adaptive --mode oracle --oracle-options=$JSON --logpath $LOGDIR --problem-file $FILE  --max_planner_time 10 | tee ./oracle/$RUN.log
done