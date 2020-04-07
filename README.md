# Color-Segmentation
Color Segmentation using GMM
============================

In this project, we have implemented an approach for robust color segmentation which was further used to detect a diferent colour objects. The different color representations of objects variations in illumination, occlusion and tilt. This is the reason Gaussian Mixture Models was used to represent these variations.

* Python 2.7, OpenCV 3.2

# Expectation-Maximization Algorithm
The expectation maximization algorithm is used to find out the mean, variances and weights in the of the different Gaussian Models that represent the buoys in the training images.


# Run the code

Enter the following to run the code.

```
python3 
em.py
detectbuoygreen.py
detectbuoyorange.py
detectbuoyyellow.py
mulgaussiandetect.py
```


# Result:

Output video showing detection of green buoys.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/55011289/78616251-bef29e00-7841-11ea-928a-f6ff94a33482.gif">
</p>


Output video showing detection of orange buoys. 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/55011289/78616035-1ba18900-7841-11ea-8324-b5b94c8646e6.gif">
</p>


Output video showing detection of yellow buoys. 

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/55011289/78615701-3a535000-7840-11ea-8ae6-ae6d87a3639f.gif">
</p>

