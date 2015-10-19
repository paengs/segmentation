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


if __name__ == '__main__' :
	
	pascal = pascal_db('val','2012','/data/PASCAL/VOCdevkit/', 'cls')
	#arch = 'segmentation/stride_32_only_label'
	arch = 'segmentation/cls_net'
	deploy = '../models/' + arch + '/test.prototxt'
	#model  = '../models/' + arch + '/finetuned_models/alex_ift_iter_1500.caffemodel'
	model  = '../models/' + arch + '/finetuned_models/ftmodels_iter_3000.caffemodel'
	net = net_wrapper(deploy, model,3)
	#net = analyzer(deploy, model)
	for i, ind in enumerate(pascal._image_index) :
		image_path = os.path.join( pascal._data_path, 'JPEGImages', ind + pascal._image_ext )
		net.run_forward(image_path)
		res = net._output['prob']
		#res = net._output['pool-global']
		res = res.squeeze()
		res = res[np.newaxis,:]
		if i == 0 :
			final_res = np.copy(res)
		else :
			final_res = np.vstack((final_res,res))
		print '{:d} th image ... {:s}'.format(i+1, ind)
	# wrote the results
	pascal.write_voc_results_files(final_res)
	pascal.do_matlab_eval('comp1')
