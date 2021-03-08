import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
	def __init__(self, input_size, output_size, kernel):
		super(BasicBlock, self).__init__()
		padding = (kernel - 1) // 2
		self.convolutional = nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=1, padding=padding, bias=False)
		self.batchnorm = nn.BatchNorm2d(output_size)
		self.ReLU = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.convolutional(x)
		x = self.batchnorm(x)
		x = self.ReLU(x)

		return x

class EncBlock(nn.Module):
	def __init__(self, input_size, output_size, kernel):
		super(EncBlock, self).__init__()
		padding = (kernel - 1) // 2
		self.convolutional = nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=1, padding=padding, bias=False)
		self.batchnorm = nn.BatchNorm2d(output_size)

	def forward(self, x):
		output = self.convolutional(x)
		output = self.batchnorm(output)

		return  torch.cat([x, output], dim=1)

class GlobalAveragePooling(nn.Module):
	def __init__(self):
		super(GlobalAveragePooling, self).__init__()

	def forward(self, feature):
		nchannel = feature.size(1)
		return F.avg_pool2d(feature, (feature.size(2), feature.size(3))).view(-1, nchannel)

class NetworkInNetwork(nn.Module):
	def __init__(self, _input_channel=3, _stage=3, _use_avg_on_conv3=True):
		super(NetworkInNetwork, self).__init__()

		input_channels = _input_channel
		stages = _stage
		use_avg_on_conv3 = _use_avg_on_conv3

		nchannels = 192
		nchannels2 = 160
		nchannels3 = 96

		blocks = [nn.Sequential() for i in range(stages)]
		blocks2 = [nn.Sequential() for i in range(stages - 2)]

		blocks[0].add_module("Block1_ConvB1", BasicBlock(input_channels, nchannels, 5))
		blocks[0].add_module("Block1_ConvB2", BasicBlock(nchannels, nchannels2, 1))
		blocks[0].add_module("Block1_ConvB3", BasicBlock(nchannels2, nchannels3, 1))
		blocks[0].add_module("Block1_MaxPool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

		blocks[1].add_module("Block2_ConvB1", BasicBlock(nchannels3, nchannels, 5))
		blocks[1].add_module("Block2_ConvB2", BasicBlock(nchannels, nchannels, 1))
		blocks[1].add_module("Block2_ConvB3", BasicBlock(nchannels, nchannels, 1))
		blocks[1].add_module("Block2_AvgPool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
		blocks[1].add_module("Block2_Encode", EncBlock(nchannels, nchannels, 1))

		blocks[2].add_module("Block3_ConvB1", BasicBlock(nchannels, nchannels, 3))
		blocks[2].add_module("Block3_ConvB2", BasicBlock(nchannels, nchannels, 1))
		blocks[2].add_module("Block3_ConvB3", BasicBlock(nchannels, nchannels, 1))

		blocks2[0].add_module("Block1_ConvB1", BasicBlock(nchannels, nchannels, 3))
		blocks2[0].add_module("Block1_ConvB2", BasicBlock(nchannels, nchannels, 1))
		blocks2[0].add_module("Block1_ConvB3", BasicBlock(nchannels, nchannels, 1))
		

		if stages > 3 and use_avg_on_conv3:
			blocks[2].add_module("Block3_AvgPool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
			blocks2[0].add_module("Block1_AvgPool", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

		for s in range(3, stages):
			blocks[s].add_module('Block'+str(s + 1)+'_ConvB1', BasicBlock(nchannels, nchannels, 3))
			blocks[s].add_module('Block'+str(s + 1)+'_ConvB2', BasicBlock(nchannels, nchannels, 1))
			blocks[s].add_module('Block'+str(s + 1)+'_ConvB3', BasicBlock(nchannels, nchannels, 1))
			
			blocks2[s - 2].add_module('Block'+str(s - 1)+'_ConvB1', BasicBlock(nchannels, nchannels, 3))
			blocks2[s - 2].add_module('Block'+str(s - 1)+'_ConvB2', BasicBlock(nchannels, nchannels, 1))
			blocks2[s - 2].add_module('Block'+str(s - 1)+'_ConvB3', BasicBlock(nchannels, nchannels, 1))


		blocks.append(nn.Sequential())
		blocks2.append(nn.Sequential())
		
		blocks[-1].add_module('GlobalAveragePooling', GlobalAveragePooling())
		blocks2[-1].add_module('GlobalAveragePooling', GlobalAveragePooling())

		self.feature_blocks = nn.ModuleList(blocks)
		self.feature_blocks2 = nn.ModuleList(blocks2)
		
		self.all_feature_names = ['conv'+str(s + 1) for s  in range(stages)] + ['classifier1']
		#self.all_feature_names2 = self.all_feature_names2

		self.weight_initialization()

	def parse_output_key_arg(self, output_feature_keys):
		output_feature_keys = [self.all_feature_names[-1],] if output_feature_keys is None else output_feature_keys
		#output_feature_keys2 = [self.all_feature_names2[-1],] if output_feature_keys2 is None else output_feature_keys

		if len(output_feature_keys) == 0:
			raise ValueError('Empty list of output feature keys.')

		for f, key in enumerate(output_feature_keys):
			if key not in self.all_feature_names:
				raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))

			elif key in output_feature_keys[:f]:
				raise ValueError('Duplicate output feature key: {0}.'.format(key))

		max_output_feature = max([self.all_feature_names.index(key) for key in output_feature_keys])
		
		return output_feature_keys, max_output_feature

	def forward(self, x, output_feature_keys=None):
		output_feature_keys, max_output_feature = self.parse_output_key_arg(output_feature_keys)
		output_features = [None] * len(output_feature_keys)

		feature = x

		for i in range(2):
			feature = self.feature_blocks[i](feature)
			key = self.all_feature_names[i]
			if key in output_feature_keys:
				output_features[output_feature_keys.index(key)] = feature

		mu = feature[:, :192]
		logvar = feature[:, 192:]
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		feature = eps.mul(std * 0.001).add_(mu)
		
		eps2 = torch.randn_like(std)
		feature2 = eps2.mul(std * 0.001).add(mu)

		for i in range(2, max_output_feature + 1):
			feature = self.feature_blocks[i](feature)
			feature2 = self.feature_blocks2[i - 2](feature2)

			key = self.all_feature_names[i]

			if key in output_feature_keys:
				output_features[output_feature_keys.index(key)] = feature


		output_features = output_features[0] if len(output_features) == 1 else output_features

		return output_features, feature2

	def weight_initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.weight.requires_grad:
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					m.weight.data.normal_(0, math.sqrt(2. / n))

			elif isinstance(m, nn.BatchNorm2d):
				if m.weight.requires_grad:
					m.weight.data.fill_(1)
				if m.bias.requires_grad:
					m.bias.data.zero_()
			
			elif isinstance(m, nn.Linear):
				if m.bias.requires_grad:
					m.bias.data.zero_()

class Regressor(nn.Module):
	def __init__(self, stages=3, use_avg_on_conv3=True, indim=384, classes=8):
		super(Regressor, self).__init__()
		self.nin = NetworkInNetwork()
		self.linear1 = nn.Linear(indim, classes)
		self.linear2 = nn.Linear(indim, classes)
		self.linear3 = nn.Linear(192, classes)
		self.linear4 = nn.Linear(192, classes)
		
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x1, x2, output_feature_keys=None):
		x1 = self.nin(x1, output_feature_keys)
		x2 = self.nin(x2, output_feature_keys)

		#print(x1[0].shape)
		#print(x1[1].shape)

		if output_feature_keys == None:
			x = torch.cat((x1[0], x2[0]), dim=1)
			return x1[0], x2[0], self.linear1(x), self.linear2(x), self.linear3(x1[1]), self.linear4(x1[1])
		else:
			return x1[0], x2[0]
