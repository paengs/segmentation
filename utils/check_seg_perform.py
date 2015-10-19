import os
import sys
sys.path.append('../') # WSL lib denpendancy
import lib
import caffe
from lib.pascal_db import pascal_db
from lib.net_wrapper import net_wrapper
from lib.analyzer import analyzer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__' :
	
	pascal = pascal_db('val','2012','/data/PASCAL/VOCdevkit/', 'seg')
	arch = 'bvlc_googlenet/default'
	deploy = '../0__MODELS/' + arch + '/test.prototxt'
	model  = '../0__MODELS/' + arch + '/ft_models/seg_voc_12_iter_2250.caffemodel'
	net = net_wrapper(deploy, model)
	#net = analyzer(deploy, model)

	for i, ind in enumerate(pascal._image_index) :
		image_path = os.path.join( pascal._data_path, 'JPEGImages', ind + pascal._image_ext )
		net.run_forward(image_path)
		#res_label = net._output['pool-global']
		res = net._output['prob']
		label_map = np.argmax(res[0],axis=0)
		out_map = np.zeros((label_map.shape[0],label_map.shape[1],3))
		for k in range(res.shape[1]) :
			#indices = np.where(label_map==k)
			#out_map[indices] = pascal._color_map[k]
			out_map[label_map==k,:] = pascal._color_map[k]
			#if k is not 0 and label_map[label_map==k].shape[0] is not 0 :
			#	print pascal._classes[k-1]
			#	print label_map[label_map==k].shape
			#if k is not 0 and label_map[label_map==k].shape[0] is not 0 :
			#	print k
		gt_path = os.path.join( pascal._data_path, 'SegmentationClass', ind + '.png' )
		im = Image.open(image_path)
		gt = Image.open(gt_path)
		# gt.size
		ax = plt.subplot(1,3,1)
		ax.imshow(im)
		ax = plt.subplot(1,3,2)
		ax.imshow(gt)
		ax = plt.subplot(1,3,3)
		res_map = Image.fromarray(np.uint8(out_map))
		res_map = res_map.resize( gt.size, Image.NEAREST )
		ax.imshow(res_map)
		#save_name = 'seg_res_weight/' + ind + '_response'+ pascal._image_ext
		#plt.savefig(save_name)
		#print '{:d} th image saved {}'.format(i, save_name)
		plt.show()
