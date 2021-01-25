import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import time
import copy
from sklearn.metrics import classification_report
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

categories = ['NORMAL','PNEUMONIA']

class Model(nn.Module):
    def __init__(self, out_classes, drop=0.5,**kwargs):
        super(Model, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, padding=0, stride=1)
        self.bn2 = nn.BatchNorm2d(6)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(6, 32, kernel_size=3, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.linear1 = nn.Linear(34944, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, out_classes)
        self.dropout = nn.Dropout(drop)
        
        self.conv = nn.Sequential(self.conv1, nn.ReLU(inplace=True), #self.bn1,\
                                   self.pool1, self.dropout, self.conv2, nn.ReLU(inplace=True),# self.bn2,\
                                   	 self.pool2, self.conv3, nn.ReLU(inplace=True),# self.bn3,
                                      self.pool3) 
        self.dense = nn.Sequential(self.linear1,self.bn3, nn.ReLU(inplace=True),self.dropout,\
                                    self.linear2, self.bn4, nn.ReLU(inplace=True), self.dropout,\
                                   	 self.linear3, nn.ReLU(inplace=True), self.dropout)
    def forward(self, x):
        # x = x.type(torch.DoubleTensor)
        # print(x.shape)
        # x.type(torch.FloatTensor)
        # x = Variable(x).double()
        bs = x.shape[0]
        x = self.conv(x)
        # print(x.shape)
        # x = self.bk1(x)
        # x = self.bk2(x)
        # x = self.bk3(x)
        # exit()
        x = x.view(bs, -1)
        output = self.dense(x)
        act = torch.nn.Softmax(dim=1)
        return act(output)

output = 2
model = Model(output).float()
model = torch.load('Model/chest_xray.model',map_location=torch.device('cpu'))
print('Model Loaded!...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

def predict(image):
	resized = cv2.resize(image,(242,330))
	resized = np.expand_dims(resized,axis=0)
	resized = np.expand_dims(resized,axis=1)
	# print(image.shape)
	image = torch.from_numpy(resized).float()
	result = model.forward(image)
	tensor = torch.argmax(result, dim=1)
	# print(result.shape)
	return 1, categories[tensor[0]]

image = cv2.imread('images/image_21062020141202.jpeg',0)
print(predict(image))
