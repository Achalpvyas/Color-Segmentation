from em import *
import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np


def gaussianRequired(image):
    red = image[:, :, 0].ravel()
    green = image[:, :, 1].ravel()
    blue = image[:, :, 2].ravel()
    fig = plt.figure()
    plt1 = fig.add_subplot(221)
    plt2 = fig.add_subplot(222)
    plt3 = fig.add_subplot(223)
    plt1.hist(red, 256, [0, 256])
    plt1.set_title('Histogram of Red channel of the image')
    plt2.hist(green, 256, [0, 256])
    plt2.set_title('Histogram of green channel of the image')
    plt3.hist(blue, 256, [0, 256])
    plt3.set_title('Histogram of blue channel of the image')
    plt.show()


def accessTrainingData(path):
    red = []
    blue = []
    green = []
    imagePath = os.path.join(path, '*g')
    imageFiles = glob.glob(imagePath)
    count = 0
    for image in imageFiles:
        img = cv2.imread(image)
        croppedImage = img[2:14, 2:14]

        # Plotting histogram to determine number of
        # gaussian distributions required
        # if count == 0:
        #     gaussianRequired(croppedImage)
        # count += 1
        red.append(croppedImage[:, :, 0])
        green.append(croppedImage[:, :, 1])
        blue.append(croppedImage[:, :, 2])

    red = np.array(red).ravel()
    green = np.array(green).ravel()
    blue = np.array(blue).ravel()
    return np.concatenate((red, blue, green), axis=0).flatten()


greenData = accessTrainingData("Data/green")
print(len(greenData))
orangeData = accessTrainingData("Data/orange")
print(len(orangeData))
yellowData = accessTrainingData("Data/yellow")
