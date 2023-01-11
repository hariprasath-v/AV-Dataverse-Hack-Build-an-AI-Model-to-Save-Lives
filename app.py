import streamlit as st
import torch
from datetime import datetime
import os
import requests
import time
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as pyplot
from git import Repo




def page1():
  image_file=st.sidebar.file_uploader("choose image file",type=['png','jpg','jpeg'])
  if image_file is not None:
     img=Image.open(image_file)
     #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
     img=np.array(img)
     #img = cv2.imdecode(img, 1)
     col1, col2 = st.columns(2)
     with col1:
       st.image(img, caption='Uploaded Image', use_column_width='always',channels='RGB')
     imgpath = os.path.join('yolov7', image_file.name)
     model=torch.hub.load(".",'custom','best.pt',source='local')
     pred = model(img)
     pred.render()  # render bbox in image
     for im in pred.ims:
         im_base64 = Image.fromarray(im)
         im_base64.save(outputpath)
     img_ = Image.open(outputpath)
     with col2:
       st.image(Img_, caption='Model Prediction(s)', use_column_width='always')
  

      

if __name__ == '__main__':
  page1()


@st.cache
def loadModel():
    start_dl = time.time()
    Repo.clone_from("https://github.com/WongKinYiu/yolov7",'yolov7')
    os.chdir('yolov7')
    yolo_model=requests.get('https://api.wandb.ai/artifactsV2/gcp-us/hari141v/QXJ0aWZhY3Q6MzA1MzkxMjgy/19d391aab7b31addc02670f6de2c975c')
    with open("best.pt", 'wb')as file:
      file.write(yolo_model.content)  
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
    
loadModel() 
