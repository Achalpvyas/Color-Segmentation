from em import *
import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import imutils
from imutils import contours


def accessTrainingData(path):
    channels = []
    imagePath = os.path.join(path, '*g')
    imageFiles = glob.glob(imagePath)
    count = 0
    for image in imageFiles:
        img = cv2.imread(image)
        img = cv2.imread(image)
        croppedImage = img[2:img.shape[0] - 2, 2:img.shape[1] - 2]
        # print(croppedImage.shape)
        # Plotting histogram to determine number of
        # gaussian distributions required
        # if count == 0:
        # gaussianRequired(croppedImage)
        # count += 1
        # channels.append(croppedImage[:, :, 0])
        # channels.append(croppedImage[:, :, 1])
        croppedImage1 = croppedImage[:, :, 1]
        croppedImage = croppedImage[:, :, 2]
        for i in range(croppedImage.shape[0]):
            channels.append(croppedImage[i, :])

        for i in range(croppedImage1.shape[0]):
            channels.append(croppedImage1[i, :])

    return [np.concatenate(channels), np.concatenate(channels)]


def detectBuoy(image, mean, sd, weights, clusters):
    images = []
    data = image[:, :, 1].ravel()
    data1 = image[:, :, 2].ravel()
    Pc_x = np.zeros((len(data), clusters))
    Pc_x1 = np.zeros((len(data1), clusters))
    combinedPc_x = np.zeros((image.shape[0] * image.shape[1], clusters))
    combinedPc_x1 = np.zeros((image.shape[0] * image.shape[1], clusters))
    for cluster in range(clusters):
        Pc_x[:, cluster:cluster + 1] = np.reshape(weights[0][cluster] * gaussian(data, mean[0][cluster], sd[0][cluster]),
                                                  (len(data), 1))
        Pc_x1[:, cluster:cluster + 1] = np.reshape(weights[1][cluster] * gaussian(data1, mean[1][cluster], sd[1][cluster]),
                                                  (len(data1), 1))

        combinedPc_x = Pc_x.sum(1)
        combinedPc_x1 = Pc_x1.sum(1)

    combinedPc_x = np.reshape(combinedPc_x, (image.shape[0], image.shape[1]))
    combinedPc_x1 = np.reshape(combinedPc_x1, (image.shape[0], image.shape[1]))
    combinedPc_x = combinedPc_x + combinedPc_x1
    combinedPc_x[combinedPc_x1 > np.max(combinedPc_x1) / 2.5] = 255
    # combinedPc_x1[combinedPc_x1 > np.max(combinedPc_x1) / 2.0] = 255
    output = np.zeros_like(image)
    output[:, :, 1] = combinedPc_x
    output[:, :, 2] = combinedPc_x
    # cv2.imshow('output', output)
    blur = cv2.medianBlur(output, 3)
    cv2.imshow('Median Blur', blur)
    edged = cv2.Canny(blur, 60, 255)
    # cv2.imshow('Canny', edged)
    cont, h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont) != 0:
        cnts_sorted = sorted(cont, key=cv2.contourArea, reverse=True)
        hull = cv2.convexHull(cnts_sorted[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        # print(radius)
        if radius > 7:
            cv2.circle(image, (int(x), int(y) - 1), int(radius + 1), (0 , 255, 255), 4)
            cv2.imshow("Detecting yellow buoy", image)
            images.append(image)
        else:
            cv2.imshow("Detecting yellow buoy", image)
            images.append(image)
    else:
        cv2.imshow("Detecting yellow buoy", image)
        images.append(image)
    return images


clusters = 2
yellowData = accessTrainingData("Data/yellow")
# print(len(yellowData[1]))
mean1, sd1, weights1 = EM(yellowData[0], clusters)
mean2, sd2, weights2 = EM(yellowData[1], clusters)
mean = [mean1, mean2]
sd = [sd1, sd2]
weights = [weights1, weights2]

# Reading and Writing Video
cap = cv2.VideoCapture('detectbuoy.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detectbuoyyellow.avi', fourcc, 5.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = detectBuoy(frame, mean, sd, weights, clusters)
    for image in output:
        out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

