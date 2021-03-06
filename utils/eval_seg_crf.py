import os
import sys
sys.path.append('./') # WSL lib denpendancy
import lib
import caffe
from lib.pascal_db import pascal_db
from lib.net_wrapper import net_wrapper
from lib.analyzer import analyzer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__' :

	gpu_id = sys.argv[1]
	arch = sys.argv[2]
	setting = sys.argv[3]
	bw = sys.argv[4]
	xystd = sys.argv[5]
	rgbstd = sys.argv[6]

	# load pascal db
	pascal = pascal_db('val','2012','/data/PASCAL/VOCdevkit/', 'seg')
	
	deploy = './0__MODELS/' + arch + '/' + setting + '/crf_params/test_crf_' + bw + '_' + xystd + '_' + rgbstd + '.prototxt'
	model  = './0__MODELS/' + arch + '/' + setting + '/ft_models/ftmodels_iter_18000.caffemodel'
	net = net_wrapper(deploy, model, int(gpu_id))

	comp_id = 'comp5'
	save_path = os.path.join(pascal._devkit_path, 'results', 'VOC' + pascal._year, 'Segmentation', comp_id + '_' + pascal._image_set + '_cls' )
	if not os.path.exists( save_path ) :
		os.makedirs( save_path )

	for i, ind in enumerate(pascal._image_index) :
		image_path = os.path.join( pascal._data_path, 'JPEGImages', ind + pascal._image_ext )
		net.run_forward(image_path)
		res = net._output['prob']
		label_map = np.argmax(res[0],axis=0)
		gt_path = os.path.join( pascal._data_path, 'SegmentationClass', ind + '.png' )
		gt = Image.open(gt_path)
		res_map = Image.fromarray(np.uint8(label_map))
		res_map = res_map.resize( gt.size, Image.NEAREST )
		save_name = os.path.join( save_path, ind + '.png' )
		res_map.save(save_name)
		#print '{:d} th image saved {}'.format(i, save_name)
	
	print '{:d} th image saved {}'.format(i, save_name)
	pascal.do_matlab_eval_seg('comp5')
