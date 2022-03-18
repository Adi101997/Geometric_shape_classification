#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
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


#Function for making training and testing directories
def make_training_testing_dir():
    new_dir1 = "Training"
    new_dir2 = "Testing"
    #Change the path for parent directory
    parent_dir = "/kaggle"
    path1 = os.path.join(parent_dir, new_dir1)
    path2 = os.path.join(parent_dir, new_dir2)
    if not os.path.exists(path1) and not os.path.exists(path2):
        os.mkdir(path1)
        os.mkdir(path2)


# In[3]:


#Function for making the train and test data
def get_train_test(dataset, labels):
    list_train_x = []
    list_test_x = []
    list_train_y = []
    list_test_y = []
    subset_data_x={}
    train_test_data_x={}
    train_test_data_y={}
    index=0
    classes = np.unique(np.array(labels)) #getting the class names
    start=0
    stop=10000
    
    for i in classes:
        path_training=os.path.join("/kaggle/Training",i)
        path_testing=os.path.join("/kaggle/Testing",i)
        if not os.path.exists(path_training) and not os.path.exists(path_testing):
            #making different directories for each class in training and test folders
            os.mkdir(path_training)
            os.mkdir(path_testing)
        
        #Dividing the dataset into subsets of different classes
        subset_data_x[i] = torch.utils.data.Subset(dataset, np.arange(start,stop))
        start+=10000
        stop+=10000
        #splitting 8000 images of each class in training set and 2000 images of each classs in test set
        train_test_data_x[i] = torch.utils.data.random_split(subset_data_x[i], [8000, 2000])
        for j in range(len(train_test_data_x[i][0])):
            torchvision.utils.save_image(transforms.ToTensor()(train_test_data_x[i][0][j][0]),
                             os.path.join(path_training,i+"_"+str(j)+".png"))
        for k in range(len(train_test_data_x[i][1])):
            torchvision.utils.save_image(transforms.ToTensor()(train_test_data_x[i][1][k][0]),
                             os.path.join(path_testing,i+"_testing"+str(k)+".png"))                             


# In[4]:


#Function for getting the labels of each image
def get_labels():
    labels=[]
    #Change the path according to your folder structure
    for file_name in os.listdir('/kaggle/input/nn-hw6-data/output'):
        labels.append(file_name.split("_")[0])
    return labels


# In[5]:


#Calling the above functions for creating the train and test dataset

#Change the path for the dataset
dataset = datasets.ImageFolder('/kaggle/input/nn-hw6-data/')
make_training_testing_dir()
labels = get_labels()
get_train_test(dataset, labels)


# In[6]:


#Class taken as reference from torch3.py
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
    
    #Function for forward propagation
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


# In[7]:


#Functions taken as reference from torch3.py

#Function for training the model
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    tot_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()

        tot_loss = tot_loss + loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), tot_loss/(batch_idx+1), 100.0*correct/((batch_idx+1)*args.batch_size)))

    print('End of Epoch: {}'.format(epoch))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        tot_loss/(len(train_loader)), 100.0*correct/(len(train_loader)*args.batch_size)))
    return tot_loss/(len(train_loader)), 100.0*correct/(len(train_loader)*args.batch_size)

#Function for testing the model
def test(args, model, device, test_loader):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
        tot_loss/(len(test_loader)), 100.0*correct/(len(test_loader)*args.test_batch_size)))
    return tot_loss/(len(test_loader)), 100.0*correct/(len(test_loader)*args.test_batch_size)


# In[8]:


#Code taken as reference from torch3.py
# Training settings
parser = argparse.ArgumentParser(description='PyTorch CNN')
parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=10, help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.92, help='Learning rate step gamma (default: 0.92)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
parser.add_argument('-f')
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Transforming the image to grayscale, converting it to tensor and normalizing it
transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
#Change the path for the training and testing folder according to your folder stucture
dataset1 = datasets.ImageFolder('/kaggle/Training/', transform=transform)
dataset2 = datasets.ImageFolder('/kaggle/Testing/', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
train_acc_list=[]
test_acc_list=[]
train_loss_list=[]
test_loss_list=[]
epochs=[]
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(args, model, device, test_loader)
    scheduler.step()
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    epochs.append(epoch)

torch.save(model.state_dict(), "0602-670099560-Chaudhry.pt")


# In[9]:


#plotting training and test loss against epochs
plt.plot(epochs, train_loss_list, color='r', label='training')
plt.plot(epochs, test_loss_list, color='g', label='test')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test loss VS Epochs")
plt.legend()
plt.show()


# In[10]:


#plotting training and test accuracy against epochs
plt.plot(epochs, train_acc_list, color='r', label='training')
plt.plot(epochs, test_acc_list, color='g', label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Test accuracy VS Epochs")
plt.legend()
plt.show()

