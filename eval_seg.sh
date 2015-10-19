
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng

nohup python utils/eval_seg.py $1 $2 $3 > 0__MODELS/$2/$3/eval.log &

