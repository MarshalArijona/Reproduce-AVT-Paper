from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from NetworkInNetwork import Regressor
from MI_estimator import *
from dataset import CIFAR10
import PIL

'''
def estimate_mi(model, z, t):
	model.eval()
	z_dev = z.float().to(device)
	t_dev = t.float().to(device)

	est = model.mi_est(z_dev, t_dev)
	
	del z_dev
	del t_dev
	
	return est

def train_estimate_mi(model, device, mi_optimizer, mi_est_learning_rate, z, t):
	z_dev = z.float().to(device)
	t_dev = t.float().to(device)

	model.train()
	model_loss = - model.mi_est(z_dev, t_dev)

	mi_optimizer.zero_grad()
	model_loss.backward(retain_graph=True)
	mi_optimizer.step()

	#del z_dev, t_dev
	#torch.cuda.empty_cache()
	return model_loss
'''
if __name__ ==  '__main__':
	DATAROOT = 'dataset_cifar10'
	WORKERS = 4
	#BATCHSIZE = 200
	BATCHSIZE = 100
	IMAGESIZE = 32
	NITER = 4500
	LR = 0.001
	BETA1 = 0.5
	CUDA = True
	NGPU = 1
	NET = ''
	OPTIMIZER = ''
	OUTF = 'AVT_IWAE'
	MANUALSEED = 2
	SHIFT = 4.0
	SHRINK = 0.8
	ENLARGE = 1.2
	LRMUL = 10.0
	DIVIDE = 1000


	#mi_est_z_dim = 8 * 8 * 192
	#mi_est_t_dim = 8 
	#mi_est_hidden_size = 1000 

	#mi_est_batch_size = 50
	#mi_est_learning_rate = 0.005
	#mi_est_training_steps = 4000
	
	try:
		os.makedirs(OUTF)
	except OSError:
		pass

	if MANUALSEED is None:
		MANUALSEED = random.randint(1, 10000)

	print("Random Seed: ", MANUALSEED)
	random.seed(MANUALSEED)
	torch.manual_seed(MANUALSEED)

	cudnn.benchmark = True

	if torch.cuda.is_available() and not CUDA:
		print("WARNING: You have a CUDA device, so you should probably run with CUDA")

	train_dataset = CIFAR10(root=DATAROOT, shift=SHIFT, scale=(SHRINK, ENLARGE), fillcolor=(128, 128, 128), download=True, resample=PIL.Image.BILINEAR,
					matrix_transform=transforms.Compose([
							transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015))
						]),
					transform_pre=transforms.Compose([
							transforms.RandomCrop(32, padding=4),
							transforms.RandomHorizontalFlip(),
						]),
					transform=transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
						])
					)

	test_dataset = CIFAR10(root=DATAROOT, shift=(SHRINK, ENLARGE), fillcolor=(128, 128, 128), download=True, train=False, resample=PIL.Image.BILINEAR, 
				   matrix_transform=transforms.Compose([
				   		transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
				   	]),
				   transform_pre=transforms.Compose([
				   		transforms.RandomCrop(32, padding=4),
				   		transforms.RandomHorizontalFlip(),
				   	]),
				   transform=transforms.Compose([
				   		transforms.ToTensor(),
				   		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
				   	])

		)

	assert train_dataset
	assert test_dataset

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS)

	device = torch.device("cuda:0" if CUDA else "cpu")
	ngpu = int(NGPU)

	network = Regressor(stages=4, use_avg_on_conv3=False).to(device)
	#mi_est_model = NWJ(mi_est_z_dim, mi_est_t_dim, mi_est_hidden_size).to(device)
	if CUDA:
		network = torch.nn.DataParallel(network, device_ids=range(ngpu))

	if NET != '':
		network.load_state_dict(torch.load(NET))

	print(network)

	criterion = nn.MSELoss()

	linear2_params = list(map(id, network.module.linear2.parameters()))
	base_params = filter(lambda p: id(p) not in linear2_params, network.parameters())

	#mi_optimizer = optim.Adam(mi_est_model.parameters(), mi_est_learning_rate)
	optimizer = optim.SGD([{"params":base_params}, {"params":network.module.linear2.parameters(), "lr":LR*LRMUL}], lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
	#optimizer = optim.Adam() 
	if OPTIMIZER != '':
		optimizer.load_state_dict(torch.load(OPTIMIZER))

	for epoch in range(NITER):
		if epoch >= 50 and epoch < 3240:
			optimizer.param_groups[0]["lr"] = LR * 5
			optimizer.param_groups[1]["lr"] = LR * LRMUL
		elif epoch >= 3240 and epoch < 3480:
			optimizer.param_groups[0]["lr"] = LR * 5 * 0.2
			optimizer.param_groups[1]["lr"] = LR * LRMUL
		elif epoch >= 3480 and epoch < 3640:
			optimizer.param_groups[0]["lr"] = LR * 5 * 0.04
			optimizer.param_groups[1]["lr"] = LR * LRMUL * 0.001
		elif epoch >= 3640 and epoch < 3800:
			optimizer.param_groups[0]["lr"] = LR * 5 * 0.008
			optimizer.param_groups[1]["lr"] = LR * LRMUL * 0.001 
		elif epoch >= 3800 and epoch < 4000:
			optimizer.param_groups[0]["lr"] = LR * 5 * 0.0016
			optimizer.param_groups[1]["lr"] = LR * LRMUL * 0.001
		elif epoch > 4000:
			optimizer.param_groups[0]["lr"] = LRMUL * 5 * 0.0016 - LR * 5 * 3e-6 * (epoch - 4000)
			optimizer.param_groups[1]["lr"] = LR * LRMUL * 0.001

		for i, data, in enumerate(train_loader, 0):
			network.zero_grad()
			img1 = data[0].to(device)
			img2 = data[1].to(device)
			matrix = data[2].to(device)
			matrix = matrix.view(-1, 8)

			batch_size = img1.size(0)
			
			f1, f2, output_mu, output_logvar, output_mu2, output_logvar2 = network(img1, img2)
			#sample_z = f2_sample.view(batch_size, mi_est_z_dim)	
			output_logvar = output_logvar / DIVIDE
			std_sqr = torch.exp(output_logvar)
			output_logvar2 = output_logvar2 / DIVIDE
			std_sqr2 = torch.exp(output_logvar2)

			#sample t
			#std_t = torch.exp(output_logvar * 0.5)
			#eps_t = torch.rand_like(std_sqr)
			#sample_t = eps_t.mul(std_t * 0.001).add_(output_mu)
			#sample_t = sample_t.view(batch_size, mi_est_t_dim)
              
			err_matrix = criterion(output_mu, matrix)
			#err = (torch.sum(output_logvar) + torch.sum((output_mu - matrix) ** 2 / (std_sqr + 1e-4))) / batch_size
			log_qtgz = (torch.sum(output_logvar) + torch.sum((output_mu - matrix) ** 2 / (std_sqr + 1e-4))) / batch_size
			#nwj_est = -1.0 * nwj_est(sample_z, sample_t)
			log_qtgz2 = (torch.sum(output_logvar2) + torch.sum((output_mu2 - matrix) ** 2 / (std_sqr2 + 1e-4))) / batch_size 
			entropy_t = -0.5 * torch.log(torch.sum(std_sqr2))

			err = log_qtgz + log_qtgz2 + entropy_t
			
			err.backward()
			optimizer.step()
			#mi_optimizer.step()
			
			del img1, img2
			#del img1, img2 
			torch.cuda.empty_cache()

			print('[%d/%d][%d/%d] Loss: %.4f, Loss_matrix: %.4f'
              % (epoch, NITER, i, len(train_loader),
                 err.item(), err_matrix.item()))

		if epoch % 100 == 99:
			torch.save(network.state_dict(), '%s/network_epoch_%d.pth' % (OUTF, epoch))
			torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (OUTF, epoch))