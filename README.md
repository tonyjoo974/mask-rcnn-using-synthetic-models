# mask-rcnn-using-synthetic-data

## Abstract
The durability of construction materials is mainly controlled by the rate of absorption of water by cement pastes. Therefore, it is necessary to find a way to accurately measure and estimate the water absorption as a function of time. However, the traditional approach via ASTM C1585 standard test method makes it not only difficult to estimate but also fails to provide accurate results as it can only estimate by weighing the sample at specific time intervals.

As a result, we present a new approach to water absorption estimation by leveraging image segmentation with Mask R-CNN to effectively estimate the water absorption of cement pastes accurately. In addition, we study a new approach to image segmentation. Prior work on image segmentation mainly uses manually annotated natural images as training data. Instead, we seek to find whether synthetic models can partially (i.e. combination of real and synthetic image dataset) or fully replace natural images for an image segmentation task for measuring water absorption in cement pastes. The automatically generated synthetic images can significantly reduce the overhead of manual annotation and provide more accurate boundaries to be fed into Mask R-CNN.

## Introduction 
Statistical learning methods have been utilized in the last decades to automate processes and reduce extensive human-based efforts. Specifically, machine learning and deep learning approcahes are implemented to automate processes by which algorithmic models are developed and trained using specific inputs and datasets. Supervised learning refers to training models whereas the dataset used is labeled using ground-truths. That is, for each training example input, the ground-truth label is known. To automate certain processes through a machine learning pipeline that uses image-processing techniques, the first step is to detect regions of interests (ROIs) in images that relate to the objects that are intended to be detected. 

The widely used classifier in deep learning for the purpose of image-processing is Convolutional Neural Networks (CNNs) that classifies whether a certain object is found in the image or not. Moreover, a Regional Convolutional Neural Network (R-CNN) utilizes CNNs to localize ROIs of objects to be detected. Finally, Masked R-CNN is a pixel-wise image segmentation network that classifies each pixel to a certain class and then segments pixels of the same class using a unique mask. The input dataset to Mask R-CNN is images that contain objects of interest and associated with them are ground-truth labels represented in the form of a mask. Masks are created manually through annotation tools, and then are feeded to the Mask R-CNN to train a model to automatically detect those objects for general images without further training. 

Nonetheless, the process of manually annotating images is a tedious and consuming prcoess and requires significant amount of time and human resources. Therefore, developing a further algorithm that annotates images and creates masks automatically will be a major breakthrough towards the process of full automation. The automatically created labels or masks are called synthetic images because they were synthesized by an algorithm without human interference or annotation. In this study, both manually annotated images and synthetic images are produced for a dataset of a cement paste specimen to train Mask R-CNN to detect water absorption level. Results are compared between manually annotated labels and synthetic labels by computing model efficiency and error for each dataset type.    

## Theory 

## Methods 

## Dataset

### Real Images 

### Syntehtic Images 

#### Real-time Measurements 

![alt-text](https://github.com/tonyjoo974/mask-rcnn-using-synthetic-models/blob/master/data/combined_gif12.gif)


## Discussion 

## Conclusions

