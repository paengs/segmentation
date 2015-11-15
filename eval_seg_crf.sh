
CAFFE_PATH=/home/paeng/projects/1__LIB/caffe-paeng
LOG="0__MODELS/vgg16/2scale_multi_deconv/crf_params/crf_search.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

w=3
xy=10
rgb=3
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb
rgb=5
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb
xy=30
rgb=3
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb
rgb=5
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb

w=5
xy=10
rgb=3
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb
rgb=5
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb
xy=30
rgb=3
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb
rgb=5
python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb


#python utils/eval_seg_crf.py 3 vgg16 2scale_multi_deconv $w $xy $rgb > 0__MODELS/vgg16/2scale_multi_deconv/crf_params/eval_$w$p$xy$p$rgb.log &
