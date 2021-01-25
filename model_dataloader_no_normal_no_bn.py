import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Variable
import time
import copy
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
# from torch.utils.tensorboard import SummaryWriter

data_dir = 'chest_xray/train/'
val_dir = 'chest_xray/val/'
categories = ['NORMAL','PNEUMONIA']

data_transform = transforms.Compose([
        # transforms.
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([242, 330]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5],
         #                    std=[0.5])
    ])
train_dataset = datasets.ImageFolder(root=data_dir, transform = data_transform)
val_dataset = datasets.ImageFolder(root=val_dir,transform = data_transform)

# print(train_dataset.class_to_idx)
# exit()    

train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4)
val_dataloader = DataLoader(val_dataset, batch_size = 8, shuffle = True, num_workers = 4)

class Model(nn.Module):
    def __init__(self, out_classes, drop=0.5,**kwargs):
        super(Model, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, padding=0, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(6, 32, kernel_size=3, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.linear1 = nn.Linear(34944, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, out_classes)
        self.dropout = nn.Dropout(drop)
        
        self.conv = nn.Sequential(self.conv1, nn.ReLU(inplace=True), #self.bn1,\
                                   self.pool1, self.dropout, self.conv2, nn.ReLU(inplace=True),# self.bn2,\
                                   	 self.pool2, self.conv3, nn.ReLU(inplace=True), #self.bn3,
                                      self.pool3) 
        self.dense = nn.Sequential(self.linear1,self.bn4, nn.ReLU(inplace=True),self.dropout,\
                                    self.linear2, self.bn5, nn.ReLU(inplace=True), self.dropout,\
                                   	 self.linear3, nn.ReLU(inplace=True), self.dropout)
    def forward(self, x):
        bs = x.size(0)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(bs, -1)
        output = self.dense(x)
        act = torch.nn.Softmax(dim=1)
        return act(output)


output = 2
model = Model(output).float()
model.train()
if torch.cuda.is_available():
    model.cuda()

print('Model Loaded!...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

num_epochs = 25

# train_set, val_dataset = torch.utils.data.random_split(train_dataloader, [round(len(train_dataloader)*0.8),(len(train_dataloader) - round(len(train_dataloader)*0.8))])
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
# print(dataloaders)
print('Dataloader Dictonary ready...')

since = time.time()
val_acc_history = []
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train() 
        else:
            model.eval()  

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                # print(outputs.shape)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                # _,labels_class = torch.max(labels,1)
                labels_class = labels

                if phase == 'train':
                    loss.backward()
                    optimizer.step()


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_class.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'val' and  epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model,save_name)
        if phase == 'val':
            val_acc_history.append(epoch_acc)

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# plt.plot(val_acc_history)
# plt.draw()
# plt.pause(3)

model.load_state_dict(best_model_wts)
torch.save(model,'Model/chest_xray_pytorch_no_normal_no_bn.model')

model.eval()
# print('Model Loaded!...')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
# criterion = nn.CrossEntropyLoss()
# model.eval()

count_total = 0
count_correct = 0

batch_size = 8

test_dir = 'chest_xray/test/'
test_dataset = datasets.ImageFolder(root=test_dir, transform = data_transform)
test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = True, num_workers = 4)
for inputs, labels in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # print(labels.shape)
    outputs = model(inputs)
    # print(outputs.shape)
    outputs = torch.argmax(outputs,1)
    # labels = torch.argmax(labels,1)
    for i in range(batch_size):
        if outputs[i] == labels[i]:
            count_correct+=1
            count_total+=1
        else:
            count_total+=1


print('testing acc is {:.2f}'.format((count_correct/count_total)*100))
