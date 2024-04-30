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

Thus feature A would match with feature D since they are closest to each other. Meanwhile, feature C remains unmatched since it is closest to A, but A is closest to D. Feature B also remains unmatched.

Notice that the feature matching process does not incorporate information about where the points are in the image. This is a huge loss of information as location of keypoints informs us quite a bit on how to match the keypoints. There are some feature matching approaches (such as optical flow) that do consider keypoint location when matching, but these methods require that the two images are visually close together.

## Advantages of SuperPoint and SuperGlue
The big advantages of SuperPoint and SuperGlue are that they address the challenges described above. SuperPoint feature detection can be deployed on a GPU, which significantly speeds up the frequency at which it can be run. Additionally, SuperPoint can be trained in a self-supervised manner on any image dataset. Thus, it can be trained on special cameras such as thermal or IR. 

SuperGlue utilizes an attentional graph neural network (GNN) after encoding both keypoint descriptors and location in images to perform feature matching. By leveraging this framework, SuperGlue better matches keypoints by considering the overall structure of the image. Also note that SuperGlue can actually be trained with ANY feature descriptor. SuperGlue, with slight modification, can be trained to use SIFT feature descriptors. However, as I will describe later, SuperPoint proves to be a foundationally superior keypoint detector to SIFT.

## Goals of this reimplementation
The central goal of my reimplementation of these papers is to see if I can optimize both SuperPoint and SuperGlue to run on a platform with very limited compute e.g. on a micro aerial vehicle (MAV). My goal was essentially to see if I could 

TODO INSERT GIF OF OPTICAL FLOW RUNNING ON MAV

1) reduce the computation time for SuperPoint,
2) improve total number of features matched correctly without sacrificing precision
3) see if I can adapt the feature matching pipeline to perform better in a specific environment without sacrificing too much general performance.
4) generally learn to implement the full training pipeline for both SuperPoint and SuperGlue (I trained SuperPoint from scratch and re-trained SuperGlue. The datasets used to train both are not publically available)

## Experiments

To address the first goal, I explored reducing the overall size of the SuperPoint architecture. I first reduced the number of CNN layers in the baseline SuperPoint architecture, which I called "SuperPoint Compact", and then I additionally reduced the dimensionality of the feature descriptors extracted by SuperPoint from 256 to 128. The overall number of trainable parameters in each architecture are shown below

INSERT IMAGE OF ARCHITECTURE PARAMS





