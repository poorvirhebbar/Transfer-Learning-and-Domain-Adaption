import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np



from util import *
from data import *
from data import getMnistDataset
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torchvision.utils import make_grid


torch.manual_seed(43)
np.random.seed(42)

def predict_dl(dl, enc,dec, clsfr):
    torch.cuda.empty_cache()
    batch_probs = []
    labels = []
    preds_list = []
    #dec_preds_list = []
    for xb, label in dl:
        label = label.cpu().detach().numpy().tolist()
        labels += label
        #dec_preds = enc(xb)
        probs = enc(xb).view(xb.shape[0], -1)
        probs = clsfr(probs)
        preds = probs.cpu().detach()
        #preds = probs.detach().numpy()
        preds = torch.max(preds,1).indices
        #preds = np.round(preds)
        #print(preds)
        #print(probs)
        probs = probs[:,1]
        
        preds = preds.tolist()
        preds_list += preds
        #probs = torch.round(probs.cpu().detach())
        probs = probs.cpu().detach().numpy().tolist()
        batch_probs += probs
    #batch_probs = torch.cat(batch_probs)
    #print(batch_probs)
    print("predictions = ",preds_list)
    return batch_probs,preds_list,labels#,dec_preds_list


def train(enc, disc, dataset, device):

	"""
	enc and disc are networks
	dataset["train"] and dataset["eval"] can be used
	device is either CPU or GPU
	"""

	# hyperparameters
	epochs = 50
	batch_size = 50
	best_score = 0
	historytrain = []
	historyeval = []

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)

	# Optimizers
	main_optim = optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=0.0002, betas=(0.5, 0.999))

	# iterate for epochs
	for epoch in range(1, epochs+1):

		enc.train()
		disc.train()

		# get the data loader
		dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)

		loss_epoch = 0.
		correct_epoch = 0

		for data, target in dataloader:

			# data.shape = [n,3,l,b]
			# target.shape = [n]

			data = data.to(device)
			target = target.to(device)
			# print(data.shape)

			# TRAIN DISCRIMINATOR

			# set gradients to zero
			main_optim.zero_grad()
			enc.zero_grad()
			disc.zero_grad()

			# get output of discriminator
			hidden = enc(data).view(data.shape[0], -1)
			# print(hidden.shape)
			out = disc(hidden)

			# calculate loss and update params
			loss = DiscLoss(out, target)
			loss.backward()
			main_optim.step()

			# get accuracy and loss_epoch
			correct = torch.sum(target == torch.argmax(out, 1))

			loss_epoch += len(data)*loss
			correct_epoch += correct


		loss_epoch = loss_epoch/len(dataset["train"])
		
		# Pretty Printing
		print("")
		print("Epoch %04d/%04d : Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
			(epoch, epochs, loss_epoch, correct_epoch, len(dataset["train"]), correct_epoch*100.0/float(len(dataset["train"]))))

		result = eval_model(enc, disc, dataset, device, "eval")
		print()
		print(result)
		eval_acc = result['accuracy']
		if best_score < eval_acc :
			best_score = eval_acc
			print('saving checkpoint for highest accuracy = ',best_score)
			save_model(enc, "enchighest.pt")
			save_model(disc, "dischighest.pt")

	return enc, disc

def main():


	# torch.cuda.empty_cache()
	dataset = getMnistDataset()

	device = get_device(True)

	conv = [1,4,8,16]
	fc = [32,16,2]
	shape = dataset["train"].shape


	enc = EncoderNetwork(conv, shape).to(device)
	disc = FullyConnectedNetwork(fc, enc.size).to(device)

	enc, disc = train(enc, disc, dataset, device)

	# test_dl = torch.utils.data.DataLoader(dataset["eval"], batch_size=50, shuffle=True)
    
 #    test_probs, test_preds,testy_labels = predict_dl(test_dl, enc,dec, clsfr)
    
 #    dict_save = {'Expected' : testy_labels , 'predictions' : test_preds}# , 'dec_predictions' : dec_preds}
    
 #    prediction = pd.DataFrame(dict_save, columns=['Expected','predictions']).to_csv('binary_new.csv')

 #    prediction = pd.DataFrame(test_preds, columns=['predictions']).to_csv('testpreds.csv')
 #    lr_auc = roc_auc_score(testy_labels, test_probs)

 #    print('ROC AUC=%.3f' % (lr_auc))
 #    lr_fpr, lr_tpr, _ = roc_curve(testy_labels, test_probs)
 #    plt.plot(lr_fpr, lr_tpr, marker='.', label='Binary')

 #    plt.savefig("ROC_new_bin")
 #    plt.show()

    #save_model(enc, "enc13.pt")
    #save_model(dec, "dec13.pt")
    #save_model(disc, "disc13.pt")
    #save_model(clsfr, "clsfr13.pt")

main()



main()
