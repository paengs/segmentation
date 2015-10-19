import os
import numpy as np
import xml.dom.minidom as minidom
from PIL import Image
from random import shuffle
import subprocess

class pascal_db :
	def __init__(self, image_set, year, devkit_path, task) :
		if task is not 'seg' and task is not 'cls' :
			assert 'Not Supported Task'
			
		self._year = year
		self._image_set = image_set
		self._devkit_path = devkit_path
		self._task = task
		self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
		self._color_map = self._load_voc_color_map('/home/paeng/projects/3__WSL/lib/voc_color_map.mat')
		self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
		self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
		self._image_ext = '.jpg'
		self._image_index = self._load_image_set_index()
		if self._image_set != 'test' and self._image_set != 'val' :
			shuffle(self._image_index)
		self._label_set = self._load_label_set()
		
		assert os.path.exists(self._devkit_path), \
        		'VOCdevkit path does not exist: {}'.format(self._devkit_path)
		assert os.path.exists(self._data_path), \
        		'Path does not exist: {}'.format(self._data_path)
	
	def _load_image_set_index(self) :
		"""
		Load the indexes listed in this dataset's image set file.
		"""
		if self._task is 'seg' :
			image_set_file = os.path.join(self._data_path, 'ImageSets', 'Segmentation',
    		                          self._image_set + '.txt')
		elif self._task is 'cls' :
			image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
    		                          self._image_set + '.txt')
		
		assert os.path.exists(image_set_file), \
        		'Path does not exist: {}'.format(image_set_file)
		with open(image_set_file) as f:
			image_index = [x.strip() for x in f.readlines()]
		
		return image_index

	def _load_label_set(self):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC format.
		"""
		def get_data_from_tag(node, tag):
			return node.getElementsByTagName(tag)[0].childNodes[0].data
		
		label_set = []
		for i, index in enumerate(self._image_index) :
			filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
			with open(filename) as f:
				data = minidom.parseString(f.read())

			objs = data.getElementsByTagName('object')
			num_objs = len(objs)
			assert num_objs != 0, \
					'Empty annotation file.. there is no label information in {}'.format(index)
			gt_label = np.zeros(len(self._classes), dtype=np.float)

			# Load object bounding boxes into a data frame.
			for ix, obj in enumerate(objs):
				cls = self._class_to_ind[str(get_data_from_tag(obj, "name")).lower().strip()]
				gt_label[cls] = 1

			assert np.sum(gt_label) != 0, \
					'No ground truth label in {}'.format(index)

			gt_label = gt_label/np.sum(gt_label)
			label_set.append(gt_label)
		
		return label_set
	
	def _load_voc_color_map(self, mat_color_map) :
		assert os.path.exists(mat_color_map), \
        		'VOC color map does not exist: {}'.format(mat_color_map)
		import scipy.io
		col_map = scipy.io.loadmat(mat_color_map)
		return col_map['ans']*255

	def write_voc_results_files(self, scores) :
		
		assert len(self._image_index) == scores.shape[0], \
				'Score List was wrong..'
		comp_id = 'comp1'
		path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main', comp_id + '_' )
	
		for i, cls in enumerate(self._classes) :
			filename = path + 'cls_' + self._image_set + '_' + cls + '.txt'
			with open(filename, 'wt') as f :
				for im_ind, index in enumerate(self._image_index) :
					f.write('{:s} {:.3f}\n'.format(index, scores[im_ind][i]))      # for fc ouput
					#f.write('{:s} {:.3f}\n'.format(index, scores[im_ind][i][0][0])) # for conv ouput

	def do_matlab_eval(self, comp_id, output_dir='output') :
		path = os.path.join(os.path.dirname(__file__),'VOCdevkit_matlab_wrapper')
		cmd = 'cd {} && '.format(path)
		cmd += 'matlab -nodisplay -nodesktop '
		cmd += '-r "dbstop if error; '
		cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
       			.format(self._devkit_path, comp_id, self._year, self._image_set, output_dir, 0)
		print('Running:\n{}'.format(cmd))
		status = subprocess.call(cmd, shell=True)

	def do_matlab_eval_seg(self, comp_id, output_dir='output') :
		path = os.path.join(os.path.dirname(__file__),'VOCdevkit_matlab_wrapper')
		cmd = 'cd {} && '.format(path)
		cmd += 'matlab -nodisplay -nodesktop '
		cmd += '-r "dbstop if error; '
		cmd += 'voc_eval_seg(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
       			.format(self._devkit_path, comp_id, self._year, self._image_set, output_dir)
		print('Running:\n{}'.format(cmd))
		status = subprocess.call(cmd, shell=True)

