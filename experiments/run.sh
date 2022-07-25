#!/bin/bash
set -e
EXPDIR=$1
PROBLEM_FILE_PATH=$(realpath $2)
PROBLEMS=$(ls $PROBLEM_FILE_PATH/*.yaml)
RUN_ARGS="${@:3}"
if [ -z ${SEED+x} ]
then
  EXPDIR="$EXPDIR-$SEED"
  SEED_ARG="--seed $SEED"
fi

DIR=$HOME/drake-tamp/
export CUDA_VISIBLE_DEVICES=""

mkdir -p $EXPDIR && cd $EXPDIR
for FILE in $PROBLEMS; do
  echo "Running $(realpath $FILE)"
  RUN=$(basename $FILE)
  LOGDIR=$(realpath ./$RUN)_logs/
  mkdir -p $LOGDIR && cd $LOGDIR
  timeout --signal 2 --foreground 130s python -O $DIR/experiments/main.py $RUN_ARGS --logpath $LOGDIR --problem-file $FILE --max_planner_time 30 $SEED_ARG | tee $EXPDIR/$RUN.log
  cd ..
done
