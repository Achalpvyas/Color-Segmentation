# Color-Segmentation
Color Segmentation using GMM
============================

In this project, I have implemented an approach for robust color segmentation which was further used to detect a red barrel based on shape statistics. The different color representations of red barrel contain variations in illumination, occlusion and tilt. This is the reason Gaussian Mixture Models was used to represent these variations.

* Tested on: Kubuntu 16.04.3, Intel i5-4200U (4) @ 2.600GHz 4GB
* Python 2.7, OpenCV 3.2

# Expectation-Maximization Algorithm
The expectation maximization algorithm is used to find out the mean, variances and weights in the of the different Gaussian Models that represent the byots in the training images.

The GMM is represented by -

![](images/formula.png)


