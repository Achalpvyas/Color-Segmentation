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
        # print(croppedImage[0,0])
        # print(croppedImage.shape)
        # Plotting histogram to determine number of
        # gaussian distributions required
        # if count == 0:
        # gaussianRequired(croppedImage)
        # count += 1
        # channels.append(croppedImage[:, :, 0])
        # channels.append(croppedImage[:, :, 1])
        croppedImage = croppedImage[:, :, 2]
        for i in range(croppedImage.shape[0]):
            channels.append(croppedImage[i, :])

    return np.concatenate(channels)


def detectBuoy(image, mean, sd, weights, clusters):
    images = []
    data = image[:, :, 2].ravel()
    Pc_x = np.zeros((len(data), clusters))
    combinedPc_x = np.zeros((image.shape[0] * image.shape[1], clusters))

    for cluster in range(clusters):
        Pc_x[:, cluster:cluster + 1] = np.reshape(weights[cluster] * gaussian(data, mean[cluster], sd[cluster]),
                                                  (len(data), 1))
        combinedPc_x = Pc_x.sum(1)

    combinedPc_x = np.reshape(combinedPc_x, (image.shape[0], image.shape[1]))
    combinedPc_x[combinedPc_x > np.max(combinedPc_x) / 3.5] = 255
    output = np.zeros_like(image)
    output[:, :, 2] = combinedPc_x
    blur = cv2.medianBlur(output, 5)
    cv2.imshow('Median Blur', blur)
    edged = cv2.Canny(blur, 60, 255)
    # cv2.imshow('Canny', edged)
    cont, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont) != 0:
        sorted_Contours = sorted(cont, key=cv2.contourArea, reverse=True)
        hull = cv2.convexHull(sorted_Contours[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        # print(radius)
        if radius > 3.5:

            cv2.circle(image, (int(x), int(y) - 1), int(radius + 1), (95, 174, 246), 4)

            cv2.imshow("Orange Buoy Detection", image)
            images.append(image)
        else:
            cv2.imshow("Orange Buoy Detection", image)
            images.append(image)
    else:
        cv2.imshow("Orange Buoy Detection", image)
        images.append(image)
    return images


clusters = 2
orangeData = accessTrainingData("Data/orange")
mean, sd, weights = EM(orangeData, clusters)

# Reading and writing  Video
cap = cv2.VideoCapture('detectbuoy.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detectbuoyorange.avi', fourcc, 5.0, (640, 480))
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
