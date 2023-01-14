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
     #img_arr=np.array(img)
     #img = cv2.imdecode(img, 1)
     col1, col2 = st.columns(2)
     with col1:
       st.image(img, caption='Uploaded Image', use_column_width='always',channels='RGB')
     outpath = os.path.join(os.getcwd(), f"out_{os.path.basename(image_file.name)}")
     model=torch.hub.load(".",'custom','best.pt',source='local')
     if st.button('Detect Pothole'):
        pred = model(img)
        pred.render()  # render bbox in image
        for im in pred.imgs:
          im_base64 = Image.fromarray(im)
          im_base64.save(outpath)
        img_ = Image.open(outpath)
        with col2:
           st.image(img_, caption='Model Prediction(s)', use_column_width='always',channels='RGB')
        with st.expander("View Annotation Data"):
          tab1, tab2, tab3 = st.tabs(['Pascal VOC', 'COCO','YOLO'])
          pred = model(img)
          with tab1:
            df1=pred.pandas().xyxy[0]
            st.dataframe(df1)
            st.download_button(
            label="Download Annotation Data as CSV",
            data=df1.to_csv(),
            file_name=f"Annotation(Pascal VOC) Data For {image_file.name}.csv",
            mime='text/csv')
          with tab2:
            df1=pred.pandas().xywh[0]
            st.dataframe(df1)
            st.download_button(
            label="Download Annotation Data as CSV",
            data=df1.to_csv(),
            file_name=f"Annotation(COCO) Data For {image_file.name}.csv",
            mime='text/csv')
          with tab3:
            df1=pred.pandas().xywhn[0]
            st.dataframe(df1)
            st.download_button(
            label="Download Annotation Data as CSV",
            data=df1.to_csv(),
            file_name=f"Annotation(YOLO) Data For {image_file.name}.csv",
            mime='text/csv')
  

      

if __name__ == '__main__':
  page1()


@st.cache
def loadModel():
    start_dl = time.time()
    Repo.clone_from("https://github.com/WongKinYiu/yolov7",'yolov7')
    os.chdir('yolov7')
    yolo_model=requests.get(st.secrets["yolo_model_link"])
    with open("best.pt", 'wb')as file:
      file.write(yolo_model.content)  
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
    
loadModel()
