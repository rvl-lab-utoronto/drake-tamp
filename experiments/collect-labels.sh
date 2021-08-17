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
  timeout --signal 9 --foreground 90s python -O $DIR/experiments/main.py --domain=$DOMAIN --algorithm adaptive --mode save --problem-file $FILE | tee ./save/$(basename $FILE).log
  timeout --signal 9 --foreground 90s python -O $DIR/experiments/main.py --domain=$DOMAIN --algorithm adaptive --mode oracle --problem-file $FILE | tee ./oracle/$(basename $FILE).log
done