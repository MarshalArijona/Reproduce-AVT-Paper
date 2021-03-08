from __future__ import print_function
from PIL import Image, PILLOW_VERSION
import os
import os.path
import numpy as np 
import sys
import numbers
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.transforms.functional import _get_inverse_affine_matrix

class CIFAR10(data.Dataset):
	base_folder = 'cifar-10-batches-py'
	url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	filename = "cifar-10-python.tar.gz"
	tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
	train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

	test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

	def __init__(self, root, shift=6, scale=None, resample=False, fillcolor=0, train=True,
	             transform_pre=None, transform=None, target_transform=None, matrix_transform=None,
	             download=False):

		#super(CIFAR10, self).__init__()
		self.root = root
		self.transform_pre = transform_pre
		self.transform = transform
		self.target_transform = target_transform
		self.matrix_transform = matrix_transform
		self.train = train

		if download:
			self.download_data()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
	                           ' You can use download=True to download it')

		if self.train:
			self.train_data = []
			self.train_labels = []	

			for fentry in self.train_list:
				f = fentry[0]
				file = os.path.join(self.root, self.base_folder, f)
				fo = open(file, 'rb')
				if sys.version_info[0] == 2:
					entry = pickle.load(fo)
				else:
					entry = pickle.load(fo, encoding='latin1')
				self.train_data.append(entry['data'])
				if 'labels' in entry:
					self.train_labels += entry['labels']
				else:
					self.train_labels += entry['fine_labels']
				fo.close()

			self.train_data = np.concatenate(self.train_data)
			self.train_data = self.train_data.reshape((50000, 3, 32, 32))
			self.train_data = self.train_data.transpose((0, 2, 3, 1))

		else:
			f = self.test_list[0][0]
			file = os.path.join(self.root, self.base_folder, f)
			fo = open(file, "rb")
			if sys.version_info[0] == 2:
				entry = pickle.load(fo)
			else:
				entry = pickle.load(fo, encoding='latin1')

			self.test_data = entry['data']
			
			if 'labels' in entry:
				self.test_labels = entry['labels']
			else:
				self.test_labels = entry['fine_labels']

			fo.close()
			self.test_data = self.test_data.reshape((10000, 3, 32, 32))
			self.test_data = self.test_data.transpose((0, 2, 3, 1))


		if scale is not None:
			assert isinstance(scale, (tuple, list)) and len(scale) == 2, "scale should be a list or tuple and it must be of length 2."

			for s in scale:
					if s <= 0:
						raise ValueError("scale values should be positive")

			self.scale = scale
			self.shift = shift
			self.resample = resample
			self.fillcolor = fillcolor

	@staticmethod
	def find_coeffs(pa, pb):
		matrix = []
		for p1, p2, in zip(pa, pb):
			matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
			matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

		A = np.matrix(matrix, dtype=np.float)
		B = np.array(pb).reshape(8)

		res = np.dot(np.linalg.inv(A.T * A)*A.T, B)
		return np.array(res).reshape(8)


	def __getitem__(self, index):
		if self.train:
			img1, target = self.train_data[index], self.train_labels[index]
		else:
			img1, target = self.test_data[index], self.test_labels[index]

		img1 = Image.fromarray(img1)
		if self.transform_pre is not None:
			img1 = self.transform_pre(img1)

		width, height = img1.size
		center = (img1.size[0] * 0.5 + 0.5, img1.size[1] * 0.5 + 0.5) 
		shift = [float(random.randint(-int(self.shift), int(self.shift))) for i in range(8)]
		scale = random.uniform(self.scale[0], self.scale[1])
		rotation = random.randint(0, 3)

		pts = [((0-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
			   ((width-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
			   ((width-center[0])*scale+center[0], (height-center[1])*scale+center[1]),
			   ((0-center[0])*scale+center[0], (height-center[1])*scale+center[1]),
				]

		pts = [pts[(i + rotation) % 4] for i in range(4)]
		pts = [(pts[i][0] + shift[2 * i], pts[i][1] + shift[2 * i + 1]) for i in range(4)]

		coeffs = self.find_coeffs(pts, [(0, 0), (width, 0), (width, height), (0, height)])

		kwargs = {"fillcolor" : self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
		img2 = img1.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)

		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)

		if self.target_transform is not None:
			target = self.target_transform(target)

		coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)

		if self.matrix_transform is not None:
			coeffs = self.matrix_transform(coeffs)

		return img1, img2, coeffs, target



	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)

	def _check_integrity(self):
		root = self.root
		for fentry in (self.train_list + self.test_list):
			filename, md5 = fentry[0], fentry[1]
			fpath = os.path.join(root, self.base_folder, filename)
			if not check_integrity(fpath, md5):
				return False

		return True

	def download_data(self):
		import tarfile

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		root = self.root
		download_url(self.url, root, self.filename, self.tgz_md5)

		cwd = os.getcwd()
		tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		tmp = 'train' if self.train is True else 'test'
		fmt_str += '    Split: {}\n'.format(tmp)
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str

