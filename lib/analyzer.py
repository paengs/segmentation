import os
import sys
import caffe
from .net_wrapper import net_wrapper
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

class analyzer(net_wrapper) :
	
	def __init__(self, deploy, model, on_gpu=None) :
		net_wrapper.__init__(self, deploy, model, on_gpu)
	
	def _calc_global_stride(self, layer_name):
		global_st = 1
		for layer in self._layers:
			params = [i.strip() for i in layer.split('\n')]
			for param in params:
				if 'stride' in param:
					global_st *= int(param.split()[-1])
				elif 'name' in param:
					current_layer = param.split()[-1][1:-1]
			if current_layer == layer_name :
				break
		return (current_layer, global_st)
	
	def _make_discrepancy_map_for_one_scale(self, original_response, batch, gt_classes, filter_size, stride=7) :
		#stride = filter_size/5
		num_occlusion = (batch.shape[-1]-filter_size)/stride + 1
		discrepancy_maps = np.zeros((len(gt_classes),batch.shape[-1],batch.shape[-1]))
		occluder = np.random.random((filter_size,filter_size,3))*255
		occluder = occluder.transpose((2,0,1))
		occluder = occluder - self._mean[:,np.newaxis,np.newaxis]
		for i in range(num_occlusion) :
			for j in range(num_occlusion) :
				occluded_batch = np.copy(batch)
				st_pt_x = 0 + i*stride
				end_pt_x = st_pt_x + filter_size
				st_pt_y = 0 + j*stride
				end_pt_y = st_pt_y + filter_size
				occluded_batch[:,:,st_pt_y:end_pt_y,st_pt_x:end_pt_x] = occluder
				self.run_forward_for_batch(occluded_batch)
				new_response = np.copy(self._output[self._net_output])
				for k in range(len(gt_classes)) :
					discrepancy = original_response[0,gt_classes[k]] - new_response[0,gt_classes[k]]
					if discrepancy > 0 :
						discrepancy_maps[k,st_pt_y:end_pt_y,st_pt_x:end_pt_x] += discrepancy
		return discrepancy_maps
		
	def make_discrepancy_map(self, image_path, gt_classes):
		filter_size = [ 56, 28, 14 ]
		self.run_forward(image_path)
		original_response = np.copy(self._output[self._net_output])
		if not original_response.argmax() in gt_classes :
			im_ = np.array([])
			discrepancy_maps = np.array([])
			return im_, discrepancy_maps, original_response
		# generate dense occlusion
		batch = self._load_image(image_path)
		discrepancy_maps = np.zeros((len(gt_classes),batch.shape[-1],batch.shape[-1]))
		for i in range(len(filter_size)) :
			discrepancy_maps += self._make_discrepancy_map_for_one_scale(original_response, batch, gt_classes, filter_size[i])
		
		#discrepancy_map[discrepancy_map<0] = 0
		im = Image.open(image_path)
		im = np.array(im.resize((256,256), Image.ANTIALIAS))
		image_dim = self._input_dim[2:]
		center = np.array([256,256]) / 2.0
		crop = np.tile(center, (1, 2))[0] + np.concatenate([-image_dim / 2.0, image_dim / 2.0])
		im_ = im[crop[0]:crop[2], crop[1]:crop[3],:]
		return im_, discrepancy_maps, original_response
		
	def show_activations(self, layer_name, image_path, filter_num=0):
		"""
		layer_info contains layer name and the corresponding global stride.
		Do not care if it comes from calc_global_stride.
		net should not be an initial model (forward data)
		"""
		layer_name, global_st = self._calc_global_stride(layer_name)
		self.run_forward(image_path)
		im = Image.open(image_path)
		im = np.array(im.resize((256,256), Image.ANTIALIAS))
		import itertools
		channel = 1 if im.ndim == 2 or im.shape[2] == 1 else 3
		
		# calc the corresponding convolution size
		input_dim = self._net.blobs['data'].data.shape[-1]
		target_dim = self._net.blobs[layer_name].data.shape[-1]
		filter_size = self._net.blobs[layer_name].data.shape[1]
		conv_size = input_dim - (target_dim-1) * global_st
		max_act = -9999999
		for index in range(filter_size) :
			activations = self._net.blobs[layer_name].data[0,index,:,:] # omit batch size
			curr_act = activations.max()
			if max_act < curr_act :
				max_act = curr_act
				max_filter = index
		
		activations = self._net.blobs[layer_name].data[0,max_filter,:,:]
		projected_activations = np.zeros((input_dim,input_dim))
		count_mask = np.zeros((input_dim,input_dim))
		for i in itertools.product(range(target_dim),repeat=2):
			act = activations[i]
			act_area = np.tile(act, (conv_size,conv_size))
			projected_activations[i[0]*global_st:i[0]*global_st+conv_size,i[1]*global_st:i[1]*global_st+conv_size] += act_area
			count_mask[i[0]*global_st:i[0]*global_st+conv_size,i[1]*global_st:i[1]*global_st+conv_size] += np.ones((conv_size,conv_size))
		
		projected_activations /= count_mask
		plt.figure()
		plt.imshow(im, cmap=cm.Greys_r)
		plt.imshow(projected_activations, cmap=cm.Reds, alpha=0.5)
		plt.show()
		
		return projected_activations
        
