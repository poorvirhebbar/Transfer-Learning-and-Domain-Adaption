#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms,datasets
import matplotlib.pyplot as plt

import random
import shutil

import cv2
import numpy as np
from torchvision import models

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

trainset = datasets.MNIST("", train= True, download= True, transform = transforms.Compose([transforms.ToTensor()]))
testset = datasets.MNIST("", train= False, download= True, transform = transforms.Compose([transforms.ToTensor()]))


# In[16]:


train_set = torch.utils.data.DataLoader(trainset, batch_size = len(trainset))
test_set = torch.utils.data.DataLoader(testset, batch_size = len(testset))

for data in train_set:
    X, Y = data 

for datat in test_set:
    xt, yt = datat
 


# In[3]:


def mnist_subset(subsetsize):
     # #taking 10000 images 100 of each digit
    train_set = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle= False)
    new_dataset = []
    i = 0
    count_digit = [0] * 10
    for data in train_set:
        i += 1
        if(i>subsetsize*3):
            break
        X, Y = data
        if count_digit[Y[0]] < subsetsize//10: #take 1000 images for each digit 
            count_digit[Y[0]] += 1
            new_dataset.append((X[0], Y[0]))

    print(len(new_dataset))
    print(count_digit)
    
    return new_dataset


# In[4]:


def prepare_base_train_data(Full=True, subsetsize = 0):
    if Full == True:
        return trainset
    return mnist_subset(subsetsize)


# In[5]:


def GaussianBlurAll(imgs, sigma, kernel_size=(0, 0)) -> torch.Tensor:
    """
    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    if sigma == 0:
        return imgs  # do nothing
    else:
        imgs = imgs.numpy()
        imgs_list = []
        for img in imgs:
            imgs_list.append(
                cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma)
            )
        imgs_list = np.array(imgs_list)
        imgs_list = imgs_list.transpose(0, 1, 2)
        return torch.from_numpy(imgs_list)
    


# In[6]:


def get_all_train_blurdata():
        #bluring images
    
    blur_images = GaussianBlurAll(X, 2)


    blured_dataset_train = []
    for i in range (len(trainset)):
        blured_dataset_train.append((blur_images[i].reshape(1,28,28), Y[i]))

    print("done making blured dataset for train")
    return blured_dataset_train

def get_all_test_blurdata():
    #making test dataset
    for data in test_set:
        X_test, Y_test = data  

    #making test data
    blur_images_test = GaussianBlurAll(X_test, 2)
    blured_dataset_test = []
    for i in range (len(testset)):
        blured_dataset_test.append((blur_images_test[i].reshape(1,28,28), Y_test[i]))

    print("done making blured dataset for test")
    
    return blured_dataset_test 


# In[15]:


def make_mixed_train_dataset(range_from, no_of_blur_images, no_of_base_images ):
    #making train mini dataset of about 5000 base images + 5000 blured images 

    mini_dataset = [] 
    count_base = [0] * 10
    count_blur = [0] * 10
    
    
    mid = range_from + (len(X) - range_from)//2
    
    for i in range (range_from, mid):
        if count_base[Y[i]] < no_of_blur_images//10: #take 500 images for each digit 
            count_base[Y[i]] += 1
            mini_dataset.append((X[i], Y[i]))

    blur_train = get_all_train_blurdata()
    for i in range (mid+1, len(X)):
        if count_blur[Y[i]] < no_of_base_images//10: #take 500 images for each digit 
            count_blur[Y[i]] += 1
            mini_dataset.append((blur_train[i][0], blur_train[i][1]))

    print("base",count_base)
    print("blur",count_blur)
    print("length of train data",len(mini_dataset))

    import random
    random.shuffle(mini_dataset) #shuffle the list
    
    return mini_dataset



def make_mixed_test_dataset():
    #making test dataset of 5000 base images + 5000 blur images
    test_mix = []
    count_baset = [0] * 10
    count_blurt = [0] * 10
    for i in range (int(len(xt)/2)):
        count_baset[yt[i]] += 1
        test_mix.append((xt[i], yt[i]))
    blur_test = get_all_test_blurdata()
    for i in range (int(len(blur_test)/2), len(blur_test)):
        count_blurt[yt[i]] += 1
        test_mix.append((blur_test[i][0], blur_test[i][1]))


    import random
    random.shuffle(test_mix) #shuffle the list

    print("base",count_baset)
    print("blur", count_blurt)
    print("length of test data",len(test_mix))
    
    return test_mix


# In[8]:


#range from which you will be taking samples to the last, no. of blured images in train data, no. of base images in train data(preferbly divided by 10) and mixed test data
#train_data ,test_data = make_mixed_dataset(30000, 1000, 1000)


# In[ ]:





# In[22]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
net = Net()
print(net)


# In[10]:


def load_model_param():
    
    FILE = "net.pth"
    loaded_model = Net()
    loaded_model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself
    loaded_model.eval()
    print("Loading of pre-trained parameters done")
    return loaded_model


# In[17]:


def train(trainset, file_name="", load_param=False, ):
    
    
    train_set = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    

    if load_param == True:
        model = load_model_param()
        print("model used is loaded one")
    else:
        net = Net()
        model = net
        print("model used is Net")
        
    ###############################################################Training model.
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    for epoch in range(10):
        for data in train_set:  # `data` is a batch of data
            X, y = data # X is the batch of features, y is the batch of targets.
            model.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = model(X)  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.nll_loss(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
        
    FILE = file_name
    torch.save(model.state_dict(), FILE)

    
############Calculating accuracy by testing on test data  
def train(trainset, file_name="", load_param=False, ):
    
    
    train_set = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    

    if load_param == True:
        model = load_model_param()
        print("model used is loaded one")
    else:
        net = Net()
        model = net
        print("model used is Net")
        
    ###############################################################Training model.
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    for epoch in range(10):
        for data in train_set:  # `data` is a batch of data
            X, y = data # X is the batch of features, y is the batch of targets.
            model.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = model(X)  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.nll_loss(output, y)  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!
        
    FILE = file_name
    torch.save(model.state_dict(), FILE)
    
    return model

    
############Calculating accuracy by testing on test data  
def test( testset, test_model):
    
    test_set = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)
    correct = 0
    total = 0
    
#     if loaded_model == True:
#         model = load_model_param()
#     else:
#         #net = Net()
#         model = net
        
    with torch.no_grad():
        for data in test_set:
            X, y = data
            output = test_model(X)
            #print(output)
            for idx, i in enumerate(output):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))


# In[1]:


#print(model.state_dict(), file=open("pre_tarined.txt", "+w"))


# In[23]:


def main():
    
    #train on base mnist data and save weights M1
    '''
    train_set = prepare_base_train_data(Full=False, subsetsize = 10000)
    train(train_set, "net.pth", load_param=False)
    test(testset)
    
    '''
    #making trainset for M1' and M2
    train_set = make_mixed_train_dataset(30000, 2000, 2000)
    

#     #train on base + blur data pretrained weights(M1')
#   trained_model =   train(train_set, "pre_trained_net.pth", load_param=True)
    
    
#     #testing M1' model
#     test_set = make_mixed_test_dataset() #testing on mixed data
#     print("test result on base+blur images")
#     test(test_set, trained_model)
    
#     test_set = get_all_test_blurdata() #testin on all blur data
#     print("test result on all blur images")
#     test(test_set, trained_model)
    
#     print("test result on all base images")#testing on all base images
#     test(testset, trained_model)
    
    
    
    print("M2 model training and testing")
    #train on base + blur data random weights(M2)
    trained_model = train(train_set, "random_net.pth", load_param=False)
    
    
    
    #testing M2 model
    test_set = make_mixed_test_dataset() #testing on mixed data
    print("test result on base+blur images")
    test(test_set, trained_model)
    
    test_set = get_all_test_blurdata() #testin on all blur data
    print("test result on all blur images")
    test(test_set, trained_model)
    
    print("test result on all base images")
    test(test_set, trained_model)
    

main()    
