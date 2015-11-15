import os
import sys
import re
import caffe
import numpy as np
from PIL import Image

class net_wrapper() :
	"""
	caffe wrapper for test the network
	"""
	def __init__(self, deploy_prototxt, caffemodel, on_gpu=None) :
		
		if on_gpu is not None :
			caffe.set_mode_gpu()
			caffe.set_device(on_gpu)
		
		self._net = caffe.Net(deploy_prototxt, caffemodel, caffe.TEST)
		self._net_input = self._net.inputs[0]
		self._net_output = self._net.outputs[0]
		self._layers = self._parse_deploy(deploy_prototxt)
		self._input_dim = np.array( self._net.blobs[self._net_input].data.shape )
		self._name = os.path.splitext(os.path.basename(caffemodel))[0]
		self._mean = np.array([104.008,116.669,122.675]) # BGR order ...

	def run_forward_for_batch(self, batch) :
		self._net.blobs[self._net_input].data[...] = batch
		self._output = self._net.forward() 

	def run_forward(self, image_path) :
		"""
		Read a image using PIL library (RGB order!)
		Must convert BGR order!!
		"""
		in_ = self._load_image(image_path)
		self._net.blobs[self._net_input].data[...] = in_ 
		self._output = self._net.forward() 

	def convert_to_fully_conv_deploy(self, save_file_path) :
		idx, fc_layers = self._extract_fc_layers()
		self._fc_layers = fc_layers
		# find params
		layer = self._layers[idx[0]-1]
		params = [i.strip() for i in layer.split('\n')]
		for param in params :
			if 'name' in param:
				st_layer = param.split()[-1][1:-1]
		
		with open(save_file_path,'w') as file :
			file.write("name : \"{}\"\n".format(self._name))
			file.write("input: \"{}\"\n".format(self._net_input))
			file.write("input_shape {\n")
			file.write("  dim: {:d}\n".format(self._input_dim[0]))
			file.write("  dim: {:d}\n".format(self._input_dim[1]))
			file.write("  dim: {:d}\n".format(self._input_dim[2]))
			file.write("  dim: {:d}\n".format(self._input_dim[3]))
			file.write("}\n")
			for i, layer in enumerate(self._layers) :
				if i >= idx[0] : # convert fc layer to conv layer
					params = [k.strip() for k in layer.split('\n')]
					for pidx, param in enumerate(params) :
						if 'name' in param :
							if i in idx :
								current_layer_name = param.split()[-1][1:-1]
								current_layer_name = current_layer_name + "_conv"
								file.write("  name: \"{}\"\n".format(current_layer_name) )
							else :
								file.write("  {}\n".format(param))
						elif 'top' in param :
							current_layer_name = param.split()[-1][1:-1]
							current_layer_name = current_layer_name + "_conv"
							file.write("  top: \"{}\"\n".format(current_layer_name) )
						elif 'type' in param :
							if i in idx :
								file.write("  type: \"Convolution\"\n".format(current_layer_name) )
							else :
								file.write("  {}\n".format(param))
						elif 'inner_' in param :
							file.write("  convolution_param {\n")
						elif 'num_' in param :
							file.write("    {}\n".format(param))
							if i == idx[0] : # initial fully conv : assume square kernel..
								k_size = self._net.blobs[st_layer].data.shape[-1]
								file.write("    kernel_size: {:d}\n".format(k_size))
							else :
								file.write("    kernel_size: 1\n")
						elif 'bottom' in param :
							if i == idx[0] :
								file.write("  {}\n".format(param))
							else :
								current_layer_name = param.split()[-1][1:-1]
								current_layer_name = current_layer_name + "_conv"
								file.write("  bottom: \"{}\"\n".format(current_layer_name) )
						else :
							if pidx == 0 or pidx == len(params)-2 :
								file.write("{}\n".format(param))
							else :
								file.write("  {}\n".format(param))
				else :
					file.write("{}".format(layer))
			
	def _load_image(self, image_path, batch_size=1) :
		image_dim = self._input_dim[2:]
		im = Image.open(image_path)
		im = np.array(im.resize((image_dim), Image.ANTIALIAS))
		im = im[:,:,::-1]
		im = im.transpose((2,0,1))
		im = im - self._mean[:,np.newaxis,np.newaxis]
		batch = np.zeros( (batch_size, im.shape[0], image_dim[0], image_dim[1]) ) 
		#center = np.array([256,256]) / 2.0
		#crop = np.tile(center, (1, 2))[0] + np.concatenate([-image_dim / 2.0, image_dim / 2.0])
		#im_ = im[:, crop[0]:crop[2], crop[1]:crop[3]]
		batch[0] = im
		return batch

	def _parse_deploy(self, deploy) :
		with open(deploy, 'r') as f :
			deploy_str = f.read()
		layer_pattern = 'layer[\\s\\S]+?\\n}\\n'
		layers = re.findall(layer_pattern, deploy_str)
		return layers

	def _extract_fc_layers(self):
		fc_layers = []
		idx = []
		for index, layer in enumerate(self._layers) :
			params = [i.strip() for i in layer.split('\n')]
			for param in params :
				if 'name' in param:
					current_layer_name = param.split()[-1][1:-1]
				if 'type' in param:
					current_layer_type = param.split()[-1][1:-1]
					if 'InnerProduct' in current_layer_type :
						fc_layers.append( current_layer_name )
						idx.append( index )
		return idx, fc_layers
