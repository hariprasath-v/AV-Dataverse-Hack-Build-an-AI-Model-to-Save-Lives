# AV-Dataverse-Hack-Build-an-AI-Model-to-Save-Lives

### Competition hosted on <a href="https://datahack.analyticsvidhya.com/contest/job-a-thon-november-2022/#About">Analyticsvidhya</a>

# About

### Build a computer vision-based technology to process and detect the potholes present in an image.

### This is my first object detection - computer vision hackathon. 
### From the competition starter notebook, I have tried pytorch - fasterrcnn resnet50 fpn object detection model without the proper image processing and augmentation the model performed not well and didn't learn any signals. 

### Final Competition score is 0.0060050745

### Leaderboard Rank is 72/77

### Evaluation Metric is mAP@[.5,.95]

### File information
 
 * av-dataverse-hack-build-an-ai-model-to-save-lives- EDA.ipynb [![Open in Kaggle](https://img.shields.io/static/v1?label=&message=Open%20in%20Kaggle&labelColor=grey&color=blue&logo=kaggle)](https://www.kaggle.com/hari141v/av-job-a-thon-november-2022-eda)
    #### Basic Exploratory Data Analysis
    #### Packages Used,
        * seaborn
        * Pandas
        * Numpy
        * Matplotlib
        * PIL
        * cv2
        * os
        * distance
        * imagehash
        * time
        * itertools
    #### Extract basic information about the images(width, height, color mode) and analyzed the information through visualization in the following methods.
    
    #### Total pothole wise image samples.
    ![Alt text](https://github.com/hariprasath-v/AV-Dataverse-Hack-Build-an-AI-Model-to-Save-Lives/blob/main/Exploratory%20Data%20Analysis%20Visualization%20Plots/Total%20Pothole%20Wise%20Image%20Samples.png)
    
    #### RGB color distribution analysis.
    ![Alt text](https://github.com/hariprasath-v/AV-Dataverse-Hack-Build-an-AI-Model-to-Save-Lives/blob/main/Exploratory%20Data%20Analysis%20Visualization%20Plots/Image%20and%20RGB%20Color%20Distribution.png)
 
    #### Find the similar images using different hashing algorithms.
         * Average hashing - Total matched images 284
         * Perceptual hashing - Total matched images 80
         * Difference hashing - Total matched images 80
         * Wavelet hashing - Total matched images 280
         * Color hashing - Total matched images 217120
         
    #### From the above various image hashing algorithm, the perceptual hashing, and difference hashing algorithms significantly find similar images based on the hash value.
    
    ![Alt text](https://github.com/hariprasath-v/AV-Dataverse-Hack-Build-an-AI-Model-to-Save-Lives/blob/main/Exploratory%20Data%20Analysis%20Visualization%20Plots/Image%20Similarity%20Perceptual%20Hashing.png)
          
           
     

