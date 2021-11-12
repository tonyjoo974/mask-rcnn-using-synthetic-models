# mask-rcnn-using-synthetic-data

## Project Title: Synthetic Data Generation for Mask R-CNN to Estimate Water Absorption in Cement Pastes

### Group Members:

1) [Tony Joo](https://github.com/tonyjoo974)
2) [Ahmed Ibrahim](https://github.com/Eng-Hercules) 
3) [Jordan Wu](https://github.com/jordanwu97) 
4) [Hossein Kabir](https://github.com/91105206a) 

### Current member roles and collaboration strategy

[Tony Joo](https://github.com/tonyjoo974) and [Ahmed Ibrahim](https://github.com/Eng-Hercules): Manually annotating natural/ real images (images 001a.jpg to 901a.jpg, and 001b.jpg to 901b.jpg, respectively, which can be accessed from the real.zip dataset) , optimizing the employed Mask RCNN algorithm to be compatible with our dataset, and editing results on Github.

[Jordan Wu](https://github.com/jordanwu97):  Manually annotating natural/ real images (images 001c.jpg to 901c.jpg, which can be accessed in real.zip dataset), optimizing the employed Mask RCNN algorithm to be compatible with our dataset, and developing an algorithm that replaces CVAT image annotation tool using flood_fill_helper.ipynb. 

[Hossein Kabir](https://github.com/91105206a): Manually annotating natural/real images (images 001c.jpg to 901c.jpg, which can be accessed in real.zip dataset), creating synthetic.zip image dataset (images aSlide1.jpg to dSlide225.jpg), optimizing the employed Mask RCNN algorithm to be compatible with our dataset, and defining the “water” class WaterDataset_with_its_utils.py for detecting and labelling the water level. 

## Abstract
The durability of construction materials is mainly controlled by the rate of absorption of water by cement pastes. Therefore, it is necessary to find a way to accurately measure and estimate the water absorption as a function of time. However, the traditional approach via ASTM C1585 standard test method makes it not only difficult to estimate but also fails to provide accurate results as it can only estimate by weighing the sample at specific time intervals. 

As a result, we present a new approach to water absorption estimation by leveraging image segmentation with Mask R-CNN to effectively estimate the water absorption of cement pastes accurately. In addition, we study a new approach to image segmentation. Prior work on image segmentation mainly uses manually annotated natural images as training data. Instead, we seek to find whether synthetic models can partially (i.e. combination of real and synthetic image dataset) or fully replace natural images for an image segmentation task for measuring water absorption in cement pastes. The automatically generated synthetic images can significantly reduce the overhead of manual annotation and provide more accurate boundaries to be fed into Mask R-CNN.

## Introduction 
Statistical learning methods have been utilized in the last decades to automate processes and reduce extensive human-based efforts. Specifically, machine learning and deep learning approcahes are implemented to automate processes by which algorithmic models are developed and trained using specific inputs and datasets. Supervised learning refers to training models whereas the dataset used is labeled using ground-truths. That is, for each training example input, the ground-truth label is known. To automate certain processes through a machine learning pipeline that uses image-processing techniques, the first step is to detect regions of interests (ROIs) in images that relate to the objects that are intended to be detected. 

The widely used classifier in deep learning for the purpose of image-processing is Convolutional Neural Networks (CNNs) that classifies whether a certain object is found in the image or not. Moreover, a Regional Convolutional Neural Network (R-CNN) utilizes CNNs to localize ROIs of objects to be detected. Finally, Masked R-CNN is a pixel-wise image segmentation network that classifies each pixel to a certain class and then segments pixels of the same class using a unique mask. The input dataset to Mask R-CNN is images that contain objects of interest and associated with them are ground-truth labels represented in the form of a mask. Masks are created manually through annotation tools, and then are feeded to the Mask R-CNN to train a model to automatically detect those objects for general images without further training. 

Nonetheless, the process of manually annotating images is a tedious and consuming process and requires significant amount of time and human resources. Therefore, developing a further algorithm that annotates images and creates masks automatically will be a major breakthrough towards the process of full automation. The automatically created labels or masks are called synthetic images because they were synthesized by an algorithm without human interference or annotation. In this study, both manually annotated images and synthetic images are produced for a dataset of a cement paste specimen to train Mask R-CNN to detect water absorption level. Results are compared between manually annotated labels and synthetic labels by computing model efficiency and error for each dataset type.    

## Theory 
Mask R-CNN is built upon Faster R-CNN, which proposes a method for real-time object detection with Region Proposal Networks (RPN). Mask R-CNN is a 2-stage framework, where the first stage extracts features by passing the image through a CNN backbone architecture based on ResNet, which outputs feature maps. Then, using the feature maps, RPN proposes regions by using sliding window technique on k anchor boxes. The second stage then extracts features from each candidate box using RoIAlign and it performs classification, bounding-box regression, and outputs binary masks for each RoI “in parallel” to be used for instance segmentation. 

One of the key differences of Mask R-CNN from other R-CNN is that it uses Feature Pyramid Network (FPN) for more effective backbone for feature extraction. It extracts the RoI features from different levels and the high level features are passed down to lower layers, allowing the features at every level to have access to features at lower and higher levels. 

Another key difference is that it maintains spatial structure of masks by introducing pixel-to-pixel correspondence when finding RoIs because RoIPool that was used in Faster R-CNN had to quantize a floating-number RoI to the discrete integer number of the feature map dimension, which introduced misalignments between the RoI and the extracted features. After predicting m x m mask from each RoI using an Fully-Convolutional Network, RoIAlign computes the value of each sampling point in each of 2 x 2 bins by bilinear interpolation from the nearby grid points on the feature map and this removes the quantization of the stride. Together with FCN and RoIAlign, Mask R-CNN is able to achieve pixel-to-pixel behavior, preserving spatial orientation of features with zero loss of data.

As seen from the brief analysis of Mask R-CNN above, we believe that it is the most suitable choice for our purpose in analyzing image data to identify water level from the cement paste specimens as we are estimating pixel-level water absorption.


## Methods 

### Implementing Mask- RCNN

Mask R-CNN is built upon Faster R-CNN, which proposes a method for real-time object detection with Region Proposal Networks (RPN) [1]. This method is a 2-stage framework, where the first stage extracts features by passing the image through a CNN backbone architecture based on ResNet (ResNet101 in this study), which outputs feature maps [2]. Then, using the feature maps, RPN proposes regions by using sliding window technique on k anchor boxes. The second stage then extracts features from each candidate box using RoIAlign and it performs classification, bounding-box regression, and outputs binary masks for each RoI “in parallel” to be used for instance segmentation. We also used feature activation output in the last residual stage of each stage to improve the robustness of lower-level feature extraction. Besides, the number of classes defined for our model is 2, i.e., including 1 class for background, and 1 class for “water”. The model is trained based on 20 epochs (300 iterations per epoch) with an initial learning rate of 0.001, in which the learning rate is later decreased by a factor of 10 for every 5 epochs. In addition, the model is established based on TensorFlow (version 1.15.2) and Keras (version 2.3.1) libraries, which are found to be compatible with our model. It is worth noting that the training procedure was done using NVIDIA T4 GPUs with 48 GB DDR5X memory (which can be accessed in Google Colab) and is based on the original implementation, which is currently under the MIT license [3]. Moreover, roughly 90% of the images were allocated to our training dataset, and the rest 10% were designated to the testing dataset. Finally, a detection threshold, i.e., intersection-over-union (IoU), of greater than 95% is selected, meaning that the proposals with confidence of less than 0.95 is not considered to be true positive.

### Manual Annotation of Real Images 

We realized that manual annotation of labels via CVAT is very unergonomic and time-consuming for our sample dataset. Therefore, we utilized the fact that for every frame of each specimen the borders of the subject are constant, and that only the height of the water varies. As a result, we developed an algorithm to annotate the dataset that reads the data from the stylus of any tablet (e.g., Ipad) to facilitate image annotation.

To address this issue, we first draw the boundary of the subject (in blue) as well as a reference point (in red) to flood fill from.
![](data/fast_annotation_1.png)


Then for each frame of the video, we just need to annotate the water level (in blue), making annotation much faster.
![](data/fast_annotation_2.png)

We then use the script [flood_fill_helper.ipynb](src/flood_fill_helper.ipynb) to unionize the water level annotation 
and boundary, then flood fill from the red reference point up to the bounds of the subject and water level to create our ground truth masks.
![](data/fast_annotation_mask.png)

### Synthetic Generation of Images

Our original plan was to leverage Rhinoceros software for creating synthetic image dataset; however, since it is not an open source software, we decided to simply use Microsoft Powerpoint to easily and rapidly create synthetic images as well as their corresponding masks. In the next step we manipulated (binarized, renamed, and resized) and created our dataset using binarizing_and_resizing_images.py such that it would be compatible with our Mask-RCNN model. Again, a  detailed explanation on how we created the synthetic image dataset is available on our Github.

## Dataset

Within the past 30 days, we have manually annotated 3604 real images (i.e., 901 images by each team member). These images are frames from video recordings of water absorptions by real cement pastes. Fig. 1 is an example of the image and its annotation. Specifically, as shown in the figure below, attempts have been made to generalize our model by including images that have both specular (left column) and diffuse (right column) reflections. Similarly we have created 810 synthetic images and annotations, and again, it was tried to design both specular and diffuse to better generalize our model, and to make it less sensitive to various illumination conditions. 
![alt-text](https://github.com/tonyjoo974/mask-rcnn-using-synthetic-models/blob/master/data/prog_rep_img1_u.jpg)

### Initial Results 

Considering the figure below it can be realized that the employed algorithm (trained based on synthetic data) is capable of marking the water level, even if it is applied on a video. Specifically, this figure shows the robustness of the employed method that can mark the variation of water level with time. However, the subplots shown in the figure below represent an easy dataset (without specularities) and as a result it is of interest to determine the performance of this method for analyzing difficult and complex specular images.  

![alt-text](https://github.com/tonyjoo974/mask-rcnn-using-synthetic-models/blob/master/data/prog_rep_img2.jpg)

![alt-text](https://github.com/tonyjoo974/mask-rcnn-using-synthetic-models/blob/master/data/combined_gif12.gif)

## Applying Mask RCNN on Difficult Image Data

As it was previously observed, the present Mask RCNN model (trained on synthetic dataset) is effective for analysis of simple images. However, as shown in Fig. 3a, we realize that our model can analyze complex images if the synthetic and real image datasets are merged. As a result, followed by merging the two synthetic and real dataset, we realized that our model is more accurate for analyzing difficult images, see the figure below. However, our model still fails to delineate the region of interest for super difficult images (last row of the figure) specifically if the water level is high, which is a limitation of our method, but we will try to address it in the next two weeks. 

![alt-text](https://github.com/tonyjoo974/mask-rcnn-using-synthetic-models/blob/master/data/prog_rep_img3.jpg)

## Discussion and Conclusions 

So far we have realized that our model can accurately mark the region of interest (water level) even on complex images, but we still need to model more specular synthetic images to further improve the accuracy of our model. Furthermore, we also realized that our model gives incorrect estimations whenever the water level is high. As a result, in the remaining time (next two weeks) we will dedicate our time to create models which have relatively high water levels. Besides, if time permits, we want to possibly take some time to investigate other off-the-shelf segmentation models that are less complex compared to Mask RCNN, such as U-Net. 

## References

[1] He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 2961-2969). 

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[3]  https://github.com/matterport/Mask_RCNN.git 
