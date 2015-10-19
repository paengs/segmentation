import os
import sys
sys.path.append('./caffe_for_wsl/python')
import caffe
import numpy as np


class EuclideanLossLayer(caffe.Layer):
	"""
	Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
	to demonstrate the class interface for developing layers in Python.
	"""

	def setup(self, bottom, top):
		# check input pair
		if len(bottom) != 2:
			raise Exception("Need two inputs to compute distance.")
	
	def reshape(self, bottom, top):
		# check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception("Inputs must have the same dimension.")
		# difference is shape of inputs
		self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
		# loss output is scalar
		top[0].reshape(1)

	def forward(self, bottom, top):
		import pdb
		pdb.set_trace()
		norms = np.sum( np.abs(bottom[0].data), axis=1 )
		norms = norms.reshape( norms.shape[0],1 )
		if np.isnan(norms.squeeze()).sum() != 0 :
			import pdb
			pdb.set_trace()
		#norms[norms<0.0001] = 1
		#norms = np.tile( norms, bottom[0].data.shape[1] )
		#pred = np.divide( bottom[0].data, norms )
		#self.diff[...] = pred - bottom[1].data.squeeze()
		self.diff[...] = bottom[0].data - bottom[1].data.squeeze()
		top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

	def backward(self, top, propagate_down, bottom):
		for i in range(2):
			if not propagate_down[i]:
				continue
			if i == 0:
				sign = 1
			else:
				sign = -1
			bottom[i].diff[...] = sign * self.diff / bottom[i].num
