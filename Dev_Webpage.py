import webbrowser
import streamlit as st
from datetime import datetime
from PIL import Image, ImageOps
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import cv2
import torch.nn as nn
import torch
import time
from torchvision import transforms
import torchvision.transforms.functional as TF

# from predictor import predict

CONDITION = ['NORMAL','PNEUMONIA']

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
        bs = x.shape[0]
        x = self.conv(x)
        x = x.view(bs, -1)
        output = self.dense(x)
        act = torch.nn.Sigmoid()
        return act(output)

output = 2
model = Model(output).float()
model = torch.load('Model/chest_xray_pytorch_88_backup.model',map_location=torch.device('cpu'))  #remove _[1] after training done...
print('Model Loaded!...')

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
# criterion = nn.CrossEntropyLoss()

data_transform = transforms.Compose([
        # transforms.
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([242, 330]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.5])
    ])

def predict(location):
    image = Image.open(location)
    img_tensor = data_transform(image)
    img_tensor.unsqueeze_(0)
    result = model.forward(img_tensor)
    print(result)
    tensor = torch.argmax(result, dim=1, keepdims=True)
    # print(tensor[0])
    return 1, CONDITION[tensor[0]]

header = cv2.imread('header.jpeg')
# import Model
st.image(header,use_column_width=True,channels='RGB')
st.markdown("<h1 style='text-align: center; color: skyblue; background-color: white'>Pneumonia Prediction Software</h1>", unsafe_allow_html=True)
#st.markdown("<h4 style='text-align: right; color: gray; background-color: white'>By Venkatakrishnan Sutharsan</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file(X-ray B/W image) - suggested height 242 and width 330 or similar height to width ratio..", type=["jpg","png","jpeg"])

if uploaded_file is not None:
	# st.image(uploaded_file,caption='Uploaded Image',width=500,clamp=True)
    pro_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.005)
        if i == 75:
            st.image(uploaded_file,caption='Uploaded Image',use_column_width=True,clamp=True)
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")
            print("date and time =", now.strftime("%d/%m/%Y %H:%M:%S"))
            location = 'images/image_'+dt_string+'.jpeg'
            # print(location)
            image = Image.open(uploaded_file)
            img_array = np.array(image).astype(np.float32)
            cv2.imwrite(location,img_array)
            img_array/=255.0
            print(img_array)
        pro_bar.progress(i+1)
    predicted_value, predicted = predict(location)
    if predicted_value == 1:
        st.success('Result Predicted!!!')
        if predicted == CONDITION[0]:
            st.markdown("<h4 style='text-align: center; color: green; background-color: white'>----- Result is NORMAL -----</h4>", unsafe_allow_html=True)
            st.balloons()
        elif predicted == CONDITION[1]:
            st.markdown("<h4 style='text-align: center; color: red; background-color: white'>----- Result is PNEUMONIA -----</h4>", unsafe_allow_html=True)
            # st.write(" The Result is ",predicted)
    else:
        st.error('Result Failed to Predict!!!')



# url = 'https://www.streamlit.io/'
# if st.button('Open browser'):
#     webbrowser.open_new_tab(url)