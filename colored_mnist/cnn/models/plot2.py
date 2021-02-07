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

# import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
#from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torchvision.utils import make_grid

def predict_dl(dl, enc,disc):
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
		#dec_preds = dec(dec_preds).view(xb.shape[0], -1)
		#dec_preds = clsfr(dec_preds)
		#dec_preds = torch.max(dec_preds)
		#dec_preds = torch.round(probs.cpu().detach())
		#dec_preds = dec_preds.numpy().tolist()
		#dec_preds_list += dec_preds
		#probs = dec(probs)
		probs = disc(probs)
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



def main():
	torch.cuda.empty_cache()
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	# np.random.seed(42)
	# random.seed(42)

	device = get_device(True)
	dataset = getMnistDataset()

	# device = get_device(True)
	# dataset = getBreakhisDataset()

	conv = [3,4,8,16]
	fc = [32,16,2]
	# shape = dataset["train"].shape

	PATH=""

	enc=torch.load(PATH + "enchighest.pt")
	disc=torch.load(PATH + "dischighest.pt")

	enc.eval()
	disc.eval()

	test_dl = torch.utils.data.DataLoader(dataset["eval"], batch_size=50)

	result, test_probs, test_preds,testy_labels = test_model(enc,disc, dataset, device, "eval")

	dict_save = {'Expected' : testy_labels , 'predictions' : test_preds}

	

	np.savetxt("test_labels.txt", np.hstack(testy_labels).astype(int), delimiter =',')
	np.savetxt("test_preds.txt", np.hstack(test_preds).astype(int), delimiter =',')
	arr=[]
	for i in range(len(testy_labels)):
		label=testy_labels[i]
		pred_label=test_preds[i]
		arr.append([label, pred_label])
		np.savetxt("predictions.txt", arr, delimiter =',')

	lr_auc = roc_auc_score(testy_labels, test_probs)

	print('ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
	lr_fpr, lr_tpr, _ = roc_curve(testy_labels, test_probs)
# plot the roc curve for the model
	plt.plot(lr_fpr, lr_tpr, marker='.', label='NL_nl0')
# axis labels
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
# show the legend
	plt.legend()
	plt.savefig("ROC_nl0")
# show the plot
	plt.show()

main()