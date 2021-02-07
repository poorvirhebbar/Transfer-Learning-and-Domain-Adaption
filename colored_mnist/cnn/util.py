import torch
import torch.nn as nn
import os

def get_device(cuda):
	"""
	returns the device used in the system (cpu/cuda)
	"""
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device : " + str(device))
	return device

def eval_model(enc, disc, dataset, device, folder):
	"""
	Used to evaluate a model after training.
	"""

	batch_size = 1

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)


	enc.eval()
	disc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size, shuffle=True)

	loss_epoch = 0.
	correct_epoch = 0

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		data = data.to(device)
		target = target.to(device)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)
		out = disc(hidden)

		# calculate loss and update params
		loss = DiscLoss(out, target)

		# get accuracy and loss_epoch
		correct = torch.sum(target == torch.argmax(out, 1))

		loss_epoch += len(data)*loss
		correct_epoch += correct


	loss_epoch = loss_epoch/len(dataset[folder])
	accuracy = correct_epoch*100.0/float(len(dataset[folder]))
	result = {'loss_epoch' : loss_epoch.item(), 'accuracy' : accuracy.item()}
	
	# Pretty Printing
	print("%s Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(folder, loss_epoch, correct_epoch, len(dataset[folder]), accuracy))
	return result


def save_model(model, name):
	if "models" not in os.listdir():
		os.mkdir("models")
	torch.save(model, "models/"+name)


def test_model(enc, disc, dataset, device, folder):
	"""
	Used to evaluate a model after training.
	"""

	batch_size = 1

	# Loss Functions
	DiscLoss = nn.CrossEntropyLoss().to(device)

	enc.eval()
	disc.eval()

	# get the data loader
	dataloader = torch.utils.data.DataLoader(dataset[folder], batch_size=batch_size)

	loss_epoch = 0.
	correct_epoch = 0
	batch_probs = []
	labels = []
	preds_list = []

	for data, target in dataloader:

		# data.shape = [n,3,l,b]
		# target.shape = [n]

		# put datasets on the device
		data = data.to(device)
		target = target.to(device)

		# get output of discriminator
		hidden = enc(data).view(data.shape[0], -1)
		out = disc(hidden)   #predicted output probs for two classes

		preds = out.detach().cpu()  #copy in preds
		probs = out[:,1]    #take only the probs of class1
		#print(probs)
		probs = probs.detach().cpu().numpy().tolist()
		batch_probs +=probs
		label = target.detach().cpu().numpy().tolist()
		labels+=label

		#preds = torch.round(preds.detach().cpu())
		preds = torch.max(preds,1).indices
		preds = preds.tolist()
		preds_list += preds
		loss = DiscLoss(out, target)

		# get accuracy and loss_epoch
		correct = torch.sum(target == torch.argmax(out, 1))

		loss_epoch += len(data)*loss
		correct_epoch += correct

	loss_epoch = loss_epoch/len(dataset[folder])
	accuracy = correct_epoch*100.0/float(len(dataset[folder]))
	result = {'loss_epoch' : loss_epoch.item(), 'accuracy' : accuracy.item()}
	# Pretty Printing
	print("[%s] Loss : %06f \t Accuracy : %04d/%04d (%06f)"%\
		(folder, loss_epoch, correct_epoch, len(dataset[folder]), accuracy))
	return result, batch_probs,preds_list,labels