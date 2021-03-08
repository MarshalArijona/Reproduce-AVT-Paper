import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np 
import math

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
    	super(BasicBlock, self).__init__()
    	padding = (kernel_size-1) // 2
    	self.layers = nn.Sequential()
    	self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
    	self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
    	self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

    def forward(self, x):
        print(x.shape)
        return self.layers(x)

class GlobalAveragePool(nn.Module):
	def __init__(self):
		super(GlobalAveragePool, self).__init__()

	def forward(self, x):
		x = f.avg_pool2d(x, x.size(2)).view(-1, x.size(1))
		return x

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class Classifier(nn.Module):
	def __init__(self, channel, classes, classifier_type):
		super(Classifier,self).__init__()
		NCHANNEL = channel
		NCLASSES = classes

		self.classifier_type = classifier_type
		classifier = nn.Sequential()

		if self.classifier_type == "Multilayer":
			features = min(NCLASSES*20, 2048)
			classifier.add_module("Flatten", Flatten())
			classifier.add_module("Linear1", nn.Linear(NCHANNEL, features, bias=False))
			classifier.add_module("BatchNorm1", nn.BatchNorm1d(features))
			classifier.add_module("ReLU1", nn.ReLU(inplace=True))
			classifier.add_module("Linear2", nn.Linear(features, features, bias=False))
			classifier.add_module("BatchNorm1", nn.BatchNorm1d(features))
			classifier.add_module("ReLU", nn.ReLU(inplace=False))
			classifier.add_module("Linear3", nn.Linear(features, NCLASSES))

		elif self.classifier_type == "MultilayerFC1":
			classifier.add_module("Batchnorm1", nn.BatchNorm2d(NCHANNEL // 8 // 8, affine=False))
			classifier.add_module("Flatten1", Flatten())
			classifier.add_module("Linear1", nn.Linear(NCHANNEL, NCLASSES))

		elif self.classifier_type == "MultilayerFC2":
			
			features = min(NCLASSES*20, 2048)
			classifier.add_module("Linear", Flatten())
			classifier.add_module("Linear1", nn.Linear(NCHANNEL, features, bias=False))
			classifier.add_module("Batchnorm1", nn.BatchNorm1d(features))
			classifier.add_module("ReLU1", nn.ReLU(inplace=True))
			classifier.add_module("Linear2", nn.Linear(features, NCLASSES))

		elif self.classifier_type == "NINConvolutionalBlock3":
			classifier.add_module("ConvolutionalB1", BasicBlock(NCHANNEL, 192, 3))
			classifier.add_module("ConvolutionalB2", BasicBlock(192, 192, 1))
			classifier.add_module("ConvolutionalB3", BasicBlock(192, 192, 1))
			classifier.add_module("GlbAvgPool", GlobalAveragePool())
			classifier.add_module("Linear", nn.Linear(192, NCLASSES))

		elif self.classifier_type == "AlexConvolutional4" or self.classifier_type == "AlexConvolutional5":
			if self.classifier_type == "AlexConvolutional5":
				block_5 = nn.Sequential(
						nn.Conv2d(256, 256, kernel_size = 3, padding=1),
						nn.BatchNorm2d(256),
						nn.ReLU(inplace=True),
				)
				classifier.add_module("ConvBlock5", block_5)

			classifier.add_module("Pool", nn.MaxPool2d(kernel_size=3, stride=2))
			classifier.add_module("Flatten", Flatten())
			classifier.add_module("Linear1", nn.Linear(256*6*6, 4096, bias=False))
			classifier.add_module("Batchnorm1", nn.BatchNorm1d(4096))
			classifier.add_module("ReLU1", nn.ReLU(inplace=True))
			classifier.add_module("Linear2", nn.Linear(4096, 4096, bias=False))
			classifier.add_module("Batchnorm2", nn.BatchNorm1d(4096)) 
			classifier.add_module("ReLU2", nn.ReLU(inplace=True))
			classifier.add_module("Linear3", nn.Linear(4096, NCLASSES))

		self.classifier = classifier

		self.initialize()

	def forward(self, x):
		print(x.shape)
		return self.classifier(x)

	def initialize(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					m.weight.data.fill_(1)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				fin = m.in_features
				fout = m.out_features
				std_val = np.sqrt(2.0 / fout)
				m.weight.data.normal_(0.0, std_val)
				if m.bias is not None:
					m.bias.data.fill_(0.0)
