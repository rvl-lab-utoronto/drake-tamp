#!/bin/bash
set -e
EXPDIR=$1
DOMAIN=$2
PROBLEM_FILE_PATH=$(realpath $3)
PROBLEMS=$(ls $PROBLEM_FILE_PATH/*.yaml)
DIR=$HOME/drake-tamp/
export CUDA_VISIBLE_DEVICES=""

TIMEOUT=${TIMEOUT:-90}
MAXPLAN=30
BUFFER=180
OUTER_TIMEOUT=$(($TIMEOUT + $BUFFER))

#rm -rf $EXPDIR
mkdir -p $EXPDIR && cd $EXPDIR && mkdir -p save && mkdir -p oracle
echo "timeout: $TIMEOUT" > ./collect-label-params.txt
echo "max_plan_time: $MAXPLAN" >> ./collect-label-params.txt
echo "outer_timeout: $OUTER_TIMEOUT" >> ./collect-label-params.txt
for FILE in $PROBLEMS; do
  echo "Running $FILE"
  RUN=$(basename $FILE)
  LOGDIR=$(realpath ./oracle/$RUN)_logs/
  JSON='{"data_collection_mode":true}' # don't add spaces to this or it will break
  timeout --signal 2 --foreground ${OUTER_TIMEOUT}s python -O $DIR/experiments/main.py --domain=$DOMAIN --algorithm adaptive --mode oracle --oracle-options=$JSON --logpath $LOGDIR --max-time $TIMEOUT --problem-file $FILE  --max_planner_time $MAXPLAN | tee ./oracle/$RUN.log
done





