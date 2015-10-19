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
	
	pascal = pascal_db('test','2007','/data/PASCAL/VOCdevkit/')
	
	deploy = '../models/alex/test.prototxt'
	model  = '../models/alex/finetuned_models/alex_ift_iter_1500.caffemodel'
	caffe_analyzer = analyzer(deploy, model, 3)

	for i, ind in enumerate(pascal._image_index) :
		image_path = os.path.join( pascal._data_path, 'JPEGImages', ind + pascal._image_ext )
		label = pascal._label_set[i]
		gt_classes = np.where(label!=0)[0].tolist()
		im, maps, response = caffe_analyzer.make_discrepancy_map(image_path,gt_classes)
		if im.size :
			title = ''
			for p in range(len(gt_classes)) :
				title += pascal._classes[gt_classes[p]]
				title += ': '
				title += '{:.3f}'.format(response[0,gt_classes[p]])
				title += '\n'
			for k in range(len(gt_classes)+1) :
				ax = plt.subplot(1,len(gt_classes)+1,k+1)
				if k == 0 :
					ax.imshow(im)
					ax.set_title(title)
				else :
					ax.imshow(im)
					ax.imshow(maps[k-1], alpha=0.5)
					ax.set_title(pascal._classes[gt_classes[k-1]])
			#plt.show()
			save_name = 'rf_res/' + ind + '_response'+ pascal._image_ext
			plt.savefig(save_name)
			print '{:d} th image saved {}'.format(i, save_name)
