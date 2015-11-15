#!/bin/bash

CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

export PYTHONPATH="./lib/python_layer:$CAFFE_PATH/python"

nohup $CAFFE_PATH/build/tools/caffe train -gpu $1 \
  -solver 0__MODELS/$2/$3/solver.prototxt \
  -snapshot 0__MODELS/$2/$3/ft_models/ftmodels_iter_$4.solverstate > 0__MODELS/$2/$3/experiment_resume.log &

