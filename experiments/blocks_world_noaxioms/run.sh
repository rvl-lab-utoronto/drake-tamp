#!/bin/bash
set -e
EXPDIR=$1
PROBLEM_FILE_PATH=$(realpath $2)
PROBLEMS=$(ls $PROBLEM_FILE_PATH/*.yaml)
# RUN_ARGS="${@:3}"
DIR=$HOME/drake-tamp/

mkdir -p $EXPDIR && cd $EXPDIR
for FILE in $PROBLEMS; do
  echo "Running $(realpath $FILE)"
  RUN=$(basename $FILE)
  LOGDIR=$(realpath ./$RUN)_logs/
  mkdir -p $LOGDIR && cd $LOGDIR
  timeout --signal 2 --foreground 130s python -O $DIR/lifted_search.py -t $FILE | tee $RUN.log
  cd ..
done
