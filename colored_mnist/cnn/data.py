from PIL import Image
import os
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import random



class MnistDataset(Dataset):
	
	def __init__(self, folder, noise=False, p=0):

		super(MnistDataset, self).__init__()

		self.shape = [1,7,7]
		self.num_elems = self.shape[0]*self.shape[1]*self.shape[2]
		self.num_classes = 2

		self.base = "../mnist/Output/"
        

		self.neg = os.listdir(self.base + folder + "/1") # 0
		self.pos = os.listdir(self.base + folder + "/7") # 1

		self.neg = [(self.base+folder+"/1/"+file, 0) for file in self.neg]
		self.pos = [(self.base+folder+"/7/"+file, 1) for file in self.pos]

		self.all = self.neg + self.pos
		p = (p/100.0)*len(self.all)

		if noise is True:
			choices = random.sample(range(len(self.all)), k=int(p))
			for idx in choices:
				self.all[idx] = (self.all[idx][0], (self.all[idx][1]+1)%2)

		print("About Dataset [%s]"%(folder))
		print("one: %d"%(len(self.neg)))
		print("seven: %d"%(len(self.pos)))
		print("Total: %d"%(len(self.all)))
		print("Noise: %04f"%((p*100.0)/len(self.all)))
		print("Shape: "+str(self.shape))
		print()

	def __len__(self):
		return len(self.all)

	def __getitem__(self, idx):

		img = Image.open(self.all[idx][0])
		target = self.all[idx][1]

		img = img.resize(self.shape[1:][::-1])
		trans = torchvision.transforms.ToTensor()

		return ((2.*trans(img))-1., torch.tensor(target).long())


def getMnistDataset():

	return{ "train":MnistDataset("training", True, 0.),
			"eval":MnistDataset("testing"),
			}