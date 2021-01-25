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
idx_to_class = {
    0: 'NORMAL',
    1: 'PNEUMONIA'
}

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
        
        self.conv = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.bn1,\
                                   self.pool1, self.dropout, self.conv2, nn.ReLU(inplace=True), self.bn2,\
                                     self.pool2, self.conv3, nn.ReLU(inplace=True), self.bn3,
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
model = torch.load('Model/chest_xray_pytorch2.model',map_location=torch.device('cpu'))   #_88_backup
model.eval()
print('Model Loaded!...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

data_transform = transforms.Compose([
        # transforms.
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([242, 330]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],
                             std=[0.5])
    ])


# def process_data(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (196, 196))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = img/255.0
#     img = np.reshape(img, (196,196,1))
    
#     return img

def predict(location):
    # img = process_data(location)
    # model = keras.models.load_model('Model_Keras/keras_model')
    # pred = model.predict(img)
    # y_test_hat = np.argmax(pred, axis=1)
    # print(y_test_hat)
    image = Image.open(location)
    img_tensor = data_transform(image)
    img_tensor.unsqueeze_(0)
    with torch.no_grad():
        result = model.forward(img_tensor)
    pred = torch.argmax(result, 1)
    for p in pred:
        cls = idx_to_class[p.item()]
    print(cls)
    print(result)
    # print()
    tensor = 1 if result[0][1] > result[0][0] else 0
    confidence = result[0][1] if result[0][1] > result[0][0] else result[0][0]
    # print(tensor)
    
    return 1, cls, confidence

header = cv2.imread('header.jpeg')
# import Model
st.image(header,use_column_width=True,channels='RGB')
st.markdown("<h1 style='text-align: center; color: skyblue; background-color: white'>Pneumonia Prediction Software</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: right; color: gray; background-color: white'>By Venkatakrishnan Sutharsan</h4>", unsafe_allow_html=True)

# st.markdown("<p style='text-align: center; color: red; background-color: white'>Caution: If the predicted value is less than 65 - 70% please check your doctor for accurate prediction.</p>", unsafe_allow_html=True)

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
            # print(img_array)
        pro_bar.progress(i+1)
    predicted_value, predicted, confidence = predict(location)
    if predicted_value == 1:
        st.success('Result Predicted!!!')
        if predicted == CONDITION[0]:
            st.markdown("<h4 style='text-align: center; color: green; background-color: white'>----- Result is NORMAL -----</h4>", unsafe_allow_html=True)
            st.write("Confidence is : {:.2f} %".format(confidence*100))
            st.balloons()
        elif predicted == CONDITION[1]:
            st.markdown("<h4 style='text-align: center; color: red; background-color: white'>----- Result is PNEUMONIA -----</h4>", unsafe_allow_html=True)
            st.write("Confidence is : {:.2f} %".format(confidence*100))
            # st.write(" The Result is ",predicted)
        st.markdown("<p style='text-align: center; color: red; background-color: white'>Caution: This Prediction is not intended for clinical use and consult a doctor for clinical findings...</p>", unsafe_allow_html=True)
    else:
        st.error('Result Failed to Predict!!!')



# url = 'https://www.streamlit.io/'
# if st.button('Open browser'):
#     webbrowser.open_new_tab(url)