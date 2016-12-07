#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc spatial \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
STREAM=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  sdha)
    TRAIN_IMDB="sdha_trainval"
    TEST_IMDB="sdha_test"
    PT_DIR="sdha"
    ITERS=40000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="sdha_experiments/logs/faster_rcnn_alt_opt_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --cfg sdha_experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/${STREAM}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg sdha_experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}
