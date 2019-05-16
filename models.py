import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from skimage.transform import resize
import pickle
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ADNIModel(nn.Module):
	def __init__(self, img_flag=True, ext_flag=True):
		nn.Module.__init__(self)
		self.num_ext_feats = 620906
		self.img_flag = img_flag
		self.ext_flag = ext_flag

		self.conv = nn.Sequential(
			nn.Conv3d(1, 8, 3),
			nn.ReLU(),
			nn.Conv3d(8, 8, 2),
			nn.ReLU(),
			nn.BatchNorm3d(8),
			nn.MaxPool3d(3),

			nn.Conv3d(8, 16, 3),
			nn.ReLU(),
			nn.Conv3d(16, 16, 2),
			nn.ReLU(),
			nn.BatchNorm3d(16),
			nn.MaxPool3d(3),

			nn.Conv3d(16, 32, 3),
			nn.ReLU(),
			nn.Conv3d(32, 32, 2),
			nn.ReLU(),
			nn.BatchNorm3d(32),
			nn.MaxPool3d(3),

			nn.Conv3d(32, 64, 3),
			nn.ReLU(),
			nn.Conv3d(64, 64, 2),
			nn.ReLU(),
			nn.BatchNorm3d(64),
			nn.MaxPool3d(2),
			nn.Dropout(p=0.2),
		)

		first_fc = 0
		if self.img_flag:
			first_fc += 256
		if self.ext_flag:
			first_fc += self.num_ext_feats

		self.fc = nn.Sequential(
			nn.Linear(first_fc, 128),
			nn.ReLU(),
			# nn.BatchNorm1d(),
			nn.Dropout(p=0.7),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 3),
		)

	def forward(self, img, ext):
		if self.img_flag and self.ext_flag:
			return self.forward_img_ext(img, ext)
		elif self.img_flag:
			return self.forward_img(img)
		elif self.ext_flag:
			return self.forward_ext(ext)

	def forward_img(self, img):
		out = self.conv(img)
		out = out.view(out.size(0), -1)
		return self.fc(out)

	def forward_ext(self, ext):
		return self.fc(ext)

	def forward_img_ext(self, img, ext):
		out = self.conv(img)
		out = out.view(out.size(0), -1)
		out = torch.cat((out, ext), dim=1)
		return self.fc(out)

def unpack_data(data, use_img, use_ext):
	img, ext, labels = -1, -1, -1
	if use_img and use_ext:
		img, ext, labels = data
		img, ext, labels = img.float().to(device), ext.to(device), labels.long().to(device)
	elif use_img:
		img, labels = data
		img, labels = img.float().to(device), labels.long().to(device)
	elif use_ext:
		ext, labels = data
		ext, labels = ext.to(device), labels.long().to(device)
	return img, ext, labels

if __name__ == '__main__':
	classes = ('AD', 'CN', 'MCI')

	use_img = False
	use_ext = True

	batch_size = 5
	lr = 0.005
	weight_decay = 0.2
	momentum = 0.2
	epochs = 100

	#################
	### Load Data ###
	#################
	if use_img and use_ext:
		print("Loading image and external covariate data...")
	elif use_img:
		print("Loading only image covariate data...")
	elif use_ext:
		print("Loading only external covariate data...")


	num_pickles = 4
	pickle_size = 50
	train_test_split_i = 130

	if use_img:
		with open('pickles/x0.pickle', 'rb') as f:
			img = pickle.load(f)
		for i in range(1,num_pickles):
			with open('pickles/x%d.pickle' % i, 'rb') as f:
				img = np.concatenate((img, pickle.load(f)))

		img_train = torch.Tensor(img[:train_test_split_i], device='cpu')
		img_test = torch.Tensor(img[train_test_split_i:], device='cpu')
		print("Img Train/Test:", tuple(img_train.shape),'/',tuple(img_test.shape))

	if use_ext:
		with open('pickles/ext_x.pickle', 'rb') as f:
				ext = pickle.load(f)

		ext_train = torch.Tensor(ext[:train_test_split_i], device='cpu')
		ext_test = torch.Tensor(ext[train_test_split_i:], device='cpu')
		print("Ext Train/Test:", tuple(ext_train.shape),'/',tuple(ext_test.shape))


	with open('pickles/y.pickle', 'rb') as f:
		y = pickle.load(f)

	y_train = torch.Tensor(y[:train_test_split_i], device='cpu')
	y_test = torch.Tensor(y[train_test_split_i:], device='cpu')
	print("Y Train/Test:", tuple(y_train.shape),'/',tuple(y_test.shape))

	if use_img and use_ext:
		trainset = TensorDataset(img_train, ext_train, y_train)
		testset = TensorDataset(img_test, ext_test, y_test)
	elif use_img:
		trainset = TensorDataset(img_train, y_train)
		testset = TensorDataset(img_test, y_test)
	elif use_ext:
		trainset = TensorDataset(ext_train, y_train)
		testset = TensorDataset(ext_test, y_test)

	trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
	testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

	print()

	##############
	### Model ####
	##############

	print("Loading model...")
	model = ADNIModel(img_flag=use_img, ext_flag=use_ext)
	model = model.to(device)

	print()

	################
	### Training ###
	################

	print_step = 5

	print("Training for %d epochs with lr = %.3f" % (epochs, lr))
	print()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

	for epoch in range(epochs):

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			img, ext, labels = unpack_data(data, use_img, use_ext)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(img, ext)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % print_step == 0:
				print('[%d, %2d] loss: %.5f' %
					  (epoch + 1, i + 1, running_loss / print_step))
				running_loss = 0.0
			
		correct = 0
		total = 0
		with torch.no_grad():
			for data in trainloader:
				img, ext, labels = unpack_data(data, use_img, use_ext)

				outputs = model(img, ext)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			print('Train accuracy: %.2f%%' % (100*correct/total))

		##################
		### Prediction ###
		##################

		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				img, ext, labels = unpack_data(data, use_img, use_ext)

				outputs = model(img, ext)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
			print('Test accuracy: %.2f%%' % (100*correct/total))
		print()
