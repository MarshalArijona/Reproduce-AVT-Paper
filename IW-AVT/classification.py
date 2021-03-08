from __future__ import print_function

import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from NetworkInNetwork import Regressor
from NonLinearClassifier import Classifier 

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

DATAROOT = "cifar10_classification"
WORKERS = 4
EPOCHS = 200
START_EPOCH = 0
TRAIN_BATCH = 200
TEST_BATCH = 200
LR = 0.001
SCHEDULE = [100, 150]
GAMMA = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
CHECKPOINT = "checkpoint_AVT_IWAE"
RESUME = None
NETWORK = 'network_epoch_399.pth'
MANUAL_SEED = 2
EVALUATE = None
GPU_ID = "0"

state = {"dataroot" : DATAROOT,
		 "workers" : WORKERS,
		 "epochs" : EPOCHS,
		 "start_epoch" : START_EPOCH, 
		 "train_batch" : TRAIN_BATCH,
		 "test_batch" : TEST_BATCH,
		 "lr" : LR,
		 "schedule" : SCHEDULE,
		 "gamma" : GAMMA,
		 "momentum" : MOMENTUM,
		 "weight_decay" : WEIGHT_DECAY,
		 "checkpoint" : CHECKPOINT,
		 "resume" : RESUME,
		 "network" : NETWORK,
		 "manual_seed" : MANUAL_SEED,
		 "evaluate" : EVALUATE,
		 "gpu_id" : GPU_ID}

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if MANUAL_SEED is None:
	MANUAL_SEED = random.randint(1, 10000)
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

if use_cuda:
	torch.cuda.manual_seed_all(MANUAL_SEED)

best_acc = 0

def main():
	global best_acc
	global device
	global CHECKPOINT 
	

	start_epoch = START_EPOCH

	if not os.path.isdir(CHECKPOINT):
		mkdir_p(CHECKPOINT)

	transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	dataloader = datasets.CIFAR10
	classes = 10

	trainset = dataloader(root=DATAROOT, train=True, download=True, transform=transform_train)
	trainloader = data.DataLoader(trainset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=WORKERS)

	testset = dataloader(root=DATAROOT, train=False, download=False, transform=transform_test)
	testloader = data.DataLoader(testset, batch_size=TEST_BATCH, shuffle=False, num_workers=WORKERS)

	network = Regressor(stages=4, use_avg_on_conv3=False).to(device)
	network = torch.nn.DataParallel(network, device_ids=[0])

	if NETWORK != "":
		network.load_state_dict(torch.load(NETWORK))

	model = Classifier(channel=192*8*8, classes=10, classifier_type="NINConvolutionalBlock3").to(device)
	model = torch.nn.DataParallel(model, device_ids=[0])

	cudnn.benchmark = True
	print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

	title = "cifar-10-"
	if RESUME:
		print('==> Resuming from checkpoint..')
		assert os.path.isfile(RESUME), 'Error: no checkpoint directory found!'
		
		CHECKPOINT = os.path.dirname(RESUME)
		checkpoint = torch.load(RESUME)
		best_acc = checkpoint["best_acc"]
		start_epoch = checkpoint["epoch"]
		model.load_state_dict(checkpoint["state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		logger = Logger(os.path.join(CHECKPOINT, "log.txt"), title=title, resume=True)
	else:
		logger = Logger(os.path.join(CHECKPOINT, "log.txt"), title=title)
		logger.set_names(["Learning Rate", "Train Loss", "Valid Loss", "Train Acc.", "Valid acc."])

	if EVALUATE:
		print('\nEvaluation only')
		test_loss, test_acc = test(testloader, network, model, criterion, start_epoch, use_cuda, device)
		print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
		return

	for epoch in range(start_epoch, EPOCHS):
		adjust_learning_rate(optimizer, epoch)
		
		print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, EPOCHS, state['lr']))

		train_loss, train_acc = train(trainloader, network, model, criterion, optimizer, epoch, use_cuda, device)
		test_loss, test_acc = test(testloader, network, model, criterion, epoch, use_cuda, device)

		logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

		is_best = test_acc > best_acc
		best_acc = max(test_acc, best_acc)
		save_checkpoint({
			'epoch' : epoch + 1,
			'state_dict' : model.state_dict(),
			'acc' : test_acc,
			'best_acc' : best_acc,
			'optimizer' : optimizer.state_dict(),
			}, is_best, checkpoint=CHECKPOINT)

	logger.close()
	logger.plot()
	savefig(os.path.join(CHECKPOINT, 'logs.eps'))

	print('best acc:')
	print(best_acc)


def train(trainloader, network, model, criterion, optimizer, epoch, use_cuda, device):
	network.eval()
	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()

	bar = Bar("Processing", max=len(trainloader))
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		data_time.update(time.time() - end)

		if use_cuda:
			inputs, targets = inputs.to(device), targets.to(device)

		f1, f2 = network(inputs, inputs, output_feature_keys=["conv2"])

		out_mean = f1[:, :192]
		out_var = f1[:, 192:]
		std = torch.exp(0.5 * out_var)

		features = []
		for i in range(5):
			eps = torch.rand_like(std)
			feature = eps.mul(std * 0.001).add_(out_mean)
			features.append(feature)

		feature = features[0]
		#print(feature.shape)
		for i in range(1, 5):
			feature += features[i]

		feature = feature / 5
		#print(feature.shape)
		outputs = model(feature)
		loss = criterion(outputs, targets)

		prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
	                batch=batch_idx + 1,
	                size=len(trainloader),
	                data=data_time.avg,
	                bt=batch_time.avg,
	                total=bar.elapsed_td,
	                eta=bar.eta_td,
	                loss=losses.avg,
	                top1=top1.avg,
	                top5=top5.avg,
	                )
		bar.next()

	bar.finish()
	return(losses.avg, top1.avg)

def test(testloader, network, model, criterion, epoch, use_cuda, device):
	global best_acc

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	network.eval()
	model.eval()

	end = time.time()
	bar = Bar("Processing", max=len(testloader))
	for batch_idx, (inputs, targets) in enumerate(testloader):
		data_time.update(time.time() - end)

		if use_cuda:
			inputs, targets = inputs.to(device), targets.to(device)

		f1, f2 = network(inputs, inputs, output_feature_keys=['conv2'])

		out_mean = f1[:, :192]
		output_var = f1[:, 192:]

		std = torch.exp(0.5 * output_var)
		features=[]

		for i in range(5):
			eps = torch.rand_like(std)
			feature = eps.mul(std * 0.001).add_(out_mean)
			features.append(feature)

		feature = features[0]
		for i in range(1, 5):
			feature += features[i]

		feature = feature / 5.

		outputs = model(feature)
		loss = criterion(outputs, targets)

		prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		batch_time.update(time.time() - end)
		end = time.time()

		bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
	                batch=batch_idx + 1,
	                size=len(testloader),
	                data=data_time.avg,
	                bt=batch_time.avg,
	                total=bar.elapsed_td,
	                eta=bar.eta_td,
	                loss=losses.avg,
	                top1=top1.avg,
	                top5=top5.avg,
	                )

		bar.next()
	bar.finish()
	return(losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
	filepath = os.path.join(checkpoint, filename)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))

def adjust_learning_rate(optimizer, epoch):
	global state

	if epoch in SCHEDULE:
		state["lr"] *= GAMMA
		for param_group in optimizer.param_groups:
			param_group["lr"] = state["lr"]

if __name__ == "__main__":
	main() 



