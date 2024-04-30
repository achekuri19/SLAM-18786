# SLAM-18786
## Setting up the environment
```console
foo@bar:~/software$ cd SLAM-18786/
foo@bar~/software/SLAM-18786$ git submodule init
foo@bar~/software/SLAM-18786$ git submodule update --recursive
```

To view setup on the two submodules (SuperPoint and SuperGlue) check the README of each submodule.

## Introduction
This repository contains re-implementations of the [SuperPoint](https://arxiv.org/pdf/1712.07629) and [SuperGlue](https://arxiv.org/pdf/1911.11763) papers, which are concerned with sparse feature detection and matching. 

Feature detection and matching are classical problems in computer vision, 3-D reconstruction and simultaneous localization and mapping (SLAM). By matching sparse features between sets of images, we can triangulate all these points and create a 3-D pointcloud representation of the scene directly from images, as well as an understanding of camera positioning in the scene. Examples of feature matching and a 3-D pointcloud reconstruction are below.

TODO PHOTOS OF POINTCLOUD AND FEATURE MATCHING

However, failure to properly match features can be catastrophic for any pipeline as bad data muddies the accuracy of 3D reconstruction, localization or mapping. The above photo shows an example of this, with correct feature matches shown in green, and incorrect shown in red. 

As aside, the "ground-truth" matches between two images are fundamentally unknown. There is no foolproof way to say that two points in different images are matched "correctly". Thus evaluation is done using epipolar geometry, where we essentially count the amount/ratio of feature correspondences that geometrically agree with each other. Generally, it takes time to geometrically verify all the feature matches that are inliers and outliers. Thus, having a high ratio of inliers (high precision) is ideal.

Classical computer vision has created many handcrafted solutions for this problem. By leveraging intuition on what a notable point in an image should be, classical CV has produced some great results. Probably the best-performing classical feature descriptor is SIFT, which I use for comparison to SuperPoint. However, there are a few problems with classical methods. The first is time to run. Detecting sparse feature points takes significant time, as I will discuss later in the results section. The second is the feature matching process. In general, the way feature matching works, given two sets of sparse feature descriptors, is that the features are matched by which have the closest Euclidan distance to each other.

Ex: 
Image 1 has feature descriptors A = (1, 1) and B = (1, 2)

Image 2 has feature descriptors C = (2, 1) and D = (0.5, 1)

Thus feature A would match with feature D since they are closest to each other (distance 0.5). Meanwhile, feature C remains unmatched since it is closest to A (distance = 1), but A is closest to D. Feature B also remains unmatched since its closest feature is C.

Notice that the feature matching process does not incorporate information about where the points are in the image. This is a huge loss of information as location of keypoints informs us quite a bit on how to match the keypoints. There are some feature matching approaches (such as optical flow) that do consider keypoint location when matching, but these methods require that the two images are visually close together.

## Advantages of SuperPoint and SuperGlue
The big advantages of SuperPoint and SuperGlue are that they address the challenges described above. SuperPoint feature detection can be deployed on a GPU, which significantly speeds up the frequency at which it can be run. Additionally, SuperPoint can be trained in a self-supervised manner on any image dataset. Thus, it can be trained on special cameras such as thermal or IR. 

SuperGlue utilizes an attentional graph neural network (GNN) after encoding both keypoint descriptors and location in images to perform feature matching. By leveraging this framework, SuperGlue better matches keypoints by considering the overall structure of the image. Also note that SuperGlue can actually be trained with ANY feature descriptor. SuperGlue, with slight modification, can be trained to use SIFT feature descriptors. However, as I will describe later, SuperPoint proves to be a foundationally superior keypoint detector to SIFT.

## Goals of this reimplementation
The central goal of my reimplementation of these papers is to see if I can optimize both SuperPoint and SuperGlue to run on a platform with very limited compute e.g. on a micro aerial vehicle (MAV). My goal was essentially to see if I could 

TODO INSERT GIF OF OPTICAL FLOW RUNNING ON MAV

1) reduce the computation time for SuperPoint without sacrificing performance,
2) improve the overall precision of feature matching
3) see if I can adapt the feature matching pipeline to perform better in a specific environment without sacrificing too much general performance.
4) generally learn to implement the full training pipeline for both SuperPoint and SuperGlue (I trained SuperPoint from scratch and re-trained SuperGlue. The datasets used to train both are not publically available)

## Experiments

To address the first goal, I explored reducing the overall size of the SuperPoint architecture. I first reduced the number of CNN layers in the baseline SuperPoint architecture, which I called "SuperPoint Compact", and then I additionally reduced the dimensionality of the feature descriptors extracted by SuperPoint from 256 to 128. The overall number of trainable parameters in each architecture are shown below

TODO INSERT IMAGE OF ARCHITECTURE PARAMS

To address the second goal, I explored changing the weights of the loss function to favor "consistency" of feature descriptors over how confidently SuperPoint actually detects features. The loss function, shown below, weights the _descriptor_ loss which is measured by how similar two feature descriptors are in an original image and the warped version of the images, and the _detector_ loss, which is measured by how well SuperPoint actually detected keypoints. By increasing lambda, we put more stock into the overall consistency of feature descriptors rather that feature detection, which intuitively should improve precision. I re-trained SuperPoint with lambda=0.001 rather than the lambda=0.0001 described in the original paper. 

TODO INSERT IMAGE OF LOSS FUNCTION

To address the third goal, I retrained SuperGlue using the [EUROC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#the_euroc_mav_dataset) dataset, which is visual flight data from a micro-aerial vehicle. I also made a slight modification to the loss function which I will describe later (and in my video). I evaluated its performance on both a standard evaluation dataset and specifically on the EUROC dataset to see if performance in the specific environment captured by EUROC was improved. 

TODO INSERT IMAGE OF EUROC

## Results

TODO INSERT PRECISION AND TOTAL FOR SIFT SUPERPOINT

First, the performance of SIFT and SuperPoint are shown above. Both detect similar number of inlier features, but SuperPoint is clearly higher precision. SuperPoint also runs in 28.5 ms versus 162 ms for SIFT extraction.

### Goal 1: Reducing dimensionality

TODO INSERT PRECISION AND TOTAL HISTOGRAM FOR COMPACT

As shown above, the first iteration a faster SuperPoint, SuperPoint Compact, did not produce such ideal results in terms of precision. The average precision on the same dataset was lower, although the total number of inlier features detected was similar. Even the latency of running SuperPoint Compact was similar. 

TODO INSERT PRECISION AND TOTAL HISTOGRAM FOR COMPACT V2

The second iteration, SuperPoint Compact v2, actually produced worse results. The average precision was similar to SuperPoint Compact, but much fewer keypoints were actually detected. However, this version runs twice as fast.

Finally, the overall precision, total inlier points detected, number of parameters in each architecture, and runtime are shown below.


| Architecture | Precision | # of Inliers | # Parameters | Runtime (ms) |
| :---         |     ---:      |          ---: |          ---: |          ---: |
| SuperPoint (pretrained)   | 0.367 | 49.2 | 1.3M | 28.4 |
| SuperPoint Compact  | 0.107 | 68.0 | 1.08M | 29.4 |
| SuperPoint Compact v2  | 0.113 | 46.6 | 0.73M | 15.4 |


### Goal 2: Improving precision

TODO INSERT PRECISION AND TOTAL HISTOGRAM FOR PRETRAIN VS SUPER

As described earlier, I re-trained SuperPoint with new cost function weighting to favor feature descriptor loss rather than feature detector loss. I trained with the 2017 COCO dataset (> 40,000 images) for 10 epochs with an initial learning rate of 0.0005 and a batch size of 16. The results show that there is very little difference between the two networks; the pretrained model has slightly better precision while my model has slightly more inliers detected on average. The results are tabulated below

TODO INSERT TABLE

### Goal 3: Domain-specific adaptation

TODO INSERT REPROJECTION AMBIGUITY

Finally, I attempted to retrain the SuperGlue framework to work better on the EUROC dataset without sacrificing much in the way of general capability. I also reformulated the loss function so that the "ground-truth" correspondences were more lenient. In the original paper, ground-truth correspondences are 1-to-1. That is, if feature A matches with feature B, then Feature A cannot match with feature C. However, in my formulation since A is close enough to B and C, both are considered ground-truth correspondences and the loss function tries to maximize the likelihood of both as matches.

TODO INSERT SUPERGLUE HISTOGRAMS

TODO INSERT SUPERGLUE TABLE

I varied the threshold on the confidence of feature matches between 0.2, 0.5 and 0.8 for my evaluation. The results, shown above, show fairly similar performance between the pretrained model and mine at all confidence thresholds, albeit the pretrained performs slightly better.

TODO INSERT SUPERGLUE SEQUENTIAL HISTOGRAMS

TODO INSERT SUPERGLUE SEQUENTIAL TABLE

However, when evaluating on the test set of the EUROC data (shown above), it is clear that the domain-specific training worked. While the precision is similar between the two, the total number of inliers keypoints is significantly higher, indicating the training allowed the network to be more confident about feature matches in this specific environment. The difference in results between the two networks are shown visually below.

TODO INSERT SUPERGLUE SEQUENTIAL GIF




