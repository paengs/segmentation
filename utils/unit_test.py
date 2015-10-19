import os
import sys
sys.path.append('../') # WSL lib denpendancy
import lib
import caffe
from lib.pascal_db import pascal_db
from lib.net_wrapper import net_wrapper
from lib.analyzer import analyzer

if __name__ == "__main__" :
	deploy = '../models/alex/test.prototxt'
	model  = '../models/alex/finetuned_models/alex_ift_iter_1500.caffemodel'
	image_path = '/data/PASCAL/VOCdevkit/VOC2007/JPEGImages/000022.jpg'
	'''
	net = net_wrapper(deploy, model)
	net.run_forward(image_path)
	print net._net.blobs['fc8-ft'].data
	print net._net.params['fc8-ft'][1].data
	print net._net.blobs['fc7'].data.sum()

	'''
	caffe_analyzer = analyzer(deploy, model, 3)
	caffe_analyzer.make_discrepancy_map(image_path,[14,12]) #14: person, 12: horse
	
	print caffe_analyzer._net.blobs['fc8-ft'].data
	print caffe_analyzer._net.params['fc8-ft'][1].data
	print caffe_analyzer._net.blobs['fc7'].data.sum()
