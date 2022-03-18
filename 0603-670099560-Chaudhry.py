#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
from PIL import Image
import argparse
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
import glob
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


# In[2]:


#Defining the structure of our Network class, so that we can load our model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(15488, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x)
        return x


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_model = Net()
#Change the path for the model according to your folder structure
loaded_model.load_state_dict(torch.load('/kaggle/input/model-cnn/0602-670099560-Chaudhry.pt'))
loaded_model.eval()


# In[4]:


classes = ['Circle', 'Heptagon', 'Hexagon', 
           'Nonagon', 'Octagon', 'Pentagon', 
           'Square', 'Star', 'Triangle']

transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

#Path for Validation dataset
#Change this path according to your folder structure
path_val = '/kaggle/input/validation-d/Val_data'
for i in os.listdir(path_val):
    img = Image.open(os.path.join(path_val,i))
    img_transformed = transform(img)
    img_transformed = img_transformed.unsqueeze(0)
    output = loaded_model(img_transformed)
    
    print(i,":",classes[output.argmax()])
    
    

