import os
import sys
sys.path.append('../') # WSL lib denpendancy
import lib
import caffe
from lib.net_wrapper import net_wrapper

if __name__ == "__main__" :
	
	# original network
	deploy = '../models/vgg16_conv/test.prototxt'
	model  = '../models/vgg16_conv/pre.caffemodel'
	net = net_wrapper(deploy, model)
	
	# fully conv network
	deploy = '../models/vgg16_conv/test_conv.prototxt'
	net_full_conv = net_wrapper(deploy, model)
	
	params = ['fc6', 'fc7', 'fc8']
	fc_params = {pr: (net._net.params[pr][0].data, net._net.params[pr][1].data) for pr in params}
	for fc in params:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)	

	params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
	conv_params = {pr: (net_full_conv._net.params[pr][0].data, net_full_conv._net.params[pr][1].data) for pr in params_full_conv}

	for conv in params_full_conv:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

	# converted network (fully conv. net)
	for pr, pr_conv in zip(params, params_full_conv) :
		conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
		conv_params[pr_conv][1][...] = fc_params[pr][1]
	
	save_name = '../models/vgg16_conv/pre_conv.caffemodel'
	net_full_conv._net.save(save_name)
	print 'fully conv model is saved in {}'.format(save_name)

'''
	# original network
	deploy = '../models/vgg16_conv/deploy.prototxt'
	model  = '../models/vgg16_conv/pretrained_model.caffemodel'
	net = net_wrapper(deploy, model)
	net.convert_to_fully_conv_deploy('../models/vgg16_conv/test_fully_conv.prototxt')
	deploy = '../models/vgg16_conv/deploy_conv.prototxt'
	full_conv_net = net_wrapper(deploy, model)
	
	# fc layer select & convert
	params = net._fc_layers
	params_full_conv = []
	for i in range(len(params)) :
		params_full_conv.append(params[i] + '_conv')
	
	fc_params = {pr: (net._net.params[pr][0].data, net._net.params[pr][1].data) for pr in params}
	for fc in params:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)	

	conv_params = {pr: (full_conv_net._net.params[pr][0].data, full_conv_net._net.params[pr][1].data) for pr in params_full_conv}
	for conv in params_full_conv:
		print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

	# converted network (fully conv. net)
	for pr, pr_conv in zip(params, params_full_conv) :
		conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
		conv_params[pr_conv][1][...] = fc_params[pr][1]
	
	save_name = '../models/vgg16_conv/pretrained_model_full_conv.caffemodel'
	full_conv_net._net.save(save_name)
	print 'fully conv model is saved in {}'.format(save_name)
'''
