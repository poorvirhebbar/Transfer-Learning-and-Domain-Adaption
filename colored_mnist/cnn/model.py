import torch
import torch.nn as nn

class EncoderNetwork(nn.Module):
	"""
	A fully Convolutional EncoderNetwork.
	"""
	
	def __init__(self, conv, size):
		"""
		conv =:= [3,4,8,16,32,32]
		size =:= [3,256,256]
		"""

		super(EncoderNetwork, self).__init__()

		# model parameters
		kernel_size = 3
		stride = 1
		padding = 1
		dropout_p = 0.5
		leaky_relu_slope = 0.2
		self.batchnorm = True
		self.dropout = True

		self.size = size

		# declare the layers to be used
		self.batchnorm_layers = nn.ModuleList()
		self.dropout_layer = nn.Dropout2d(dropout_p)
		self.conv_layers = nn.ModuleList()
		self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

		for i in range(len(conv)-1):
			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size, stride, padding))
			self.batchnorm_layers.append(nn.BatchNorm2d(conv[i+1]))

			self.size = [conv[i+1], ((self.size[1]+2*padding-kernel_size)//stride)+1, \
						((self.size[2]+2*padding-kernel_size)//stride)+1]


		print("Encoded Space Dimensions : "+str(self.size))

		self.size = self.size[0]*self.size[1]*self.size[2]


	def forward(self, X):

		for conv, batchnorm in zip(self.conv_layers[:-1], self.batchnorm_layers[:-1]):
			if self.batchnorm is True:
				X = self.leaky_relu(batchnorm(conv(X)))
			else:
				X = self.leaky_relu(conv(X))

			if self.dropout is True:
				X = self.dropout_layer(X)

		X = self.conv_layers[-1](X)

		return X


class FullyConnectedNetwork(nn.Module):
	"""
	A Fully Connected Network.
	"""	

	def __init__(self, fc, size):
		"""
		fc =:= [256,128,64,32,2]
		size =:= 1024
		"""

		super(FullyConnectedNetwork, self).__init__()

		# model parameters
		dropout_p = 0.5
		leaky_relu_slope = 0.2
		self.batchnorm = True
		self.dropout = True

		# declare the layers to be used
		fc = [size] + fc
		self.fc_layers = nn.ModuleList()
		self.batchnorm_layers = nn.ModuleList()
		self.dropout_layer = nn.Dropout(dropout_p)
		self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

		for i in range(len(fc)-1):
			self.fc_layers.append(nn.Linear(fc[i], fc[i+1]))
			self.batchnorm_layers.append(nn.BatchNorm1d(fc[i+1]))

	def forward(self, X):

		for fc, batchnorm in zip(self.fc_layers[:-1], self.batchnorm_layers[:-1]):
			if self.batchnorm is True:
				X = self.leaky_relu(batchnorm(fc(X)))
			else:
				X = self.leaky_relu(fc(X))

			if self.dropout is True:
				X = self.dropout_layer(X)

		X = self.fc_layers[-1](X)

		return X			


# For some dummy testing

# data = torch.randn((5,3,64,64))
# conv = [3, 4, 8]
# fc = [32,16,8,2]

# enc = EncoderNetwork(conv, list(data[0].shape))
# clsr = FullyConnectedNetwork(fc, enc.size)

# h = enc(data)
# print(h.shape)
# h = h.view(h.shape[0], -1)

# out = clsr(h)
# print(out.shape)
# print(out)