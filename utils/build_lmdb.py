import os
import sys
sys.path.append('../') # WSL lib denpendancy
import lib
import caffe
from lib.pascal_db import pascal_db
import lmdb
import numpy as np
from PIL import Image
from random import shuffle

def build_image_lmdb(db, save_path) :
	"""
	Image lmdb
	"""
	if db._task is 'cls' :
		prefix = ''
	elif db._task is 'seg' :
		prefix = '_seg'

	image_list = db._image_index
	image_path = os.path.join(db._data_path,'JPEGImages')
	lmdb_name  = save_path + 'voc_' + db._image_set + '_' + db._year + prefix + '_image_lmdb'
	in_db = lmdb.open(lmdb_name, map_size=int(1e12))
	im_sz = 320
	with in_db.begin(write=True) as in_txn :
		for in_idx, in_ in enumerate(image_list) :
			in_path = image_path + '/' + in_ + db._image_ext
			#im = np.array(Image.open(in_path)) # original size
			im = Image.open(in_path)
			im = np.array(im.resize((im_sz,im_sz), Image.ANTIALIAS))
			im = im[:,:,::-1]
			im = im.transpose((2,0,1))
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
	in_db.close()

def build_label_lmdb(db, save_path) :
	"""
	Label lmdb
	"""
	if db._task is 'cls' :
		prefix = ''
	elif db._task is 'seg' :
		prefix = '_seg'

	label_list = db._label_set
	lmdb_name  = save_path + 'voc_' + db._image_set + '_' + db._year + prefix + '_label_lmdb'
	in_db = lmdb.open(lmdb_name, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn :
		for in_idx, in_ in enumerate(label_list) :
			label = in_.reshape(1,1,len(in_))
			label_dat = caffe.io.array_to_datum(label)
			in_txn.put('{:0>10d}'.format(in_idx), label_dat.SerializeToString())
	in_db.close()

def build_label_map_lmdb(db, save_path, im_sz) :
	"""
	Segmentation Map lmdb
	"""
	image_list = db._image_index
	if db._image_set.find('aug') :
		image_path = os.path.join(db._data_path,'SegmentationClassAug')
	else :
		image_path = os.path.join(db._data_path,'SegmentationClass')
		
	lmdb_name  = save_path + 'voc_' + db._image_set + '_' + db._year + '_seg_label_map_lmdb' # + str(im_sz)
	in_db = lmdb.open(lmdb_name, map_size=int(1e12))
	with in_db.begin(write=True) as in_txn :
		for in_idx, in_ in enumerate(image_list) :
			in_path = image_path + '/' + in_ + '.png'
			#im = np.array(Image.open(in_path)) # original size
			im = Image.open(in_path)
			im = np.array(im.resize((im_sz,im_sz), Image.NEAREST))
			im = im[np.newaxis,:,:]
			im_dat = caffe.io.array_to_datum(im)
			in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
	in_db.close()

def build_seg_label_msc_lmdb(db, save_path) :
	"""
	Segmentation Image lmdb
	"""
	im_sz = [ 40, 80, 160, 320 ]
	image_list = db._seg_image_index
	image_path = os.path.join(db._data_path,'SegmentationClass')
	for i in range(len(im_sz)) :
		sz = im_sz[i]
		lmdb_name  = save_path + 'voc_' + db._image_set + '_' + db._year + '_seg_label_lmdb_scale_' + str(i)
		in_db = lmdb.open(lmdb_name, map_size=int(1e12))
		with in_db.begin(write=True) as in_txn :
			for in_idx, in_ in enumerate(image_list) :
				in_path = image_path + '/' + in_ + '.png'
				im = Image.open(in_path)
				im = np.array(im.resize((sz,sz), Image.NEAREST))
				im = im[np.newaxis,:,:]
				im_dat = caffe.io.array_to_datum(im)
				in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
		in_db.close()


def build_seg_exist_label_lmdb(db, save_path) :
	"""
	Segmentation Image lmdb
	"""
	image_list = db._seg_image_index
	image_path = os.path.join(db._data_path,'SegmentationClass')
	lmdb_name  = save_path + 'voc_' + db._image_set + '_' + db._year + '_seg_exist_label_lmdb'
	in_db = lmdb.open(lmdb_name, map_size=int(1e12))
	im_sz = 320
	grid_sz = 10
	step = im_sz/grid_sz
	with in_db.begin(write=True) as in_txn :
		for in_idx, in_ in enumerate(image_list) :
			in_path = image_path + '/' + in_ + '.png'
			#im = np.array(Image.open(in_path)) # original size
			im = Image.open(in_path)
			im = np.array(im.resize((im_sz,im_sz), Image.NEAREST))
			# grid size...
			exist_ind = 0
			exist_vec = np.zeros(grid_sz*grid_sz, dtype=np.float)
			for ii in range(grid_sz) :
				for jj in range(grid_sz) :
					patch = im[jj*step:(jj+1)*step, ii*step:(ii+1)*step]
					obj_exist = 0
					for cc in range(20) :
						if patch[patch==cc+1].shape[0] > 0 :
							obj_exist = 1
					exist_vec[exist_ind] = obj_exist
					exist_ind = exist_ind + 1
			#exist_vec = exist_vec/np.sqrt(np.sum(exist_vec)) # l2 normalization
			label = exist_vec.reshape(1,1,len(exist_vec))
			im_dat = caffe.io.array_to_datum(label)
			in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
	in_db.close()



if __name__ == '__main__' :
	lmdb_path = '../1__DATA/lmdb_seg_40/'
	if not os.path.exists( lmdb_path ) :
		os.makedirs( lmdb_path )
	
	pascal = pascal_db('train','2012','/data/PASCAL/VOCdevkit/', 'seg')
	build_image_lmdb( pascal, lmdb_path )
	print 'image lmdb build is finished...'
	build_label_lmdb( pascal, lmdb_path )
	print 'label lmdb build is finished...'
	build_label_map_lmdb( pascal, lmdb_path, 40 )
	print 'label map lmdb build is finished...'

	with open(lmdb_path + 'lmdb_path_' + pascal._image_set + '_' + pascal._year + '.txt', 'w') as file:
		for item in pascal._image_index:
			file.write("{}\n".format(item))

