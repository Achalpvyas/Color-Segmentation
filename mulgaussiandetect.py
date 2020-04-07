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
        croppedImage = img[2:img.shape[0] - 2, 2:img.shape[1] - 2]
        # print(croppedImage.shape)
        # Plotting histogram to determine number of
        # gaussian distributions required
        # if count == 0:
        # gaussianRequired(croppedImage)
        # count += 1
        # channels.append(croppedImage[:, :, 0])
        # channels.append(croppedImage[:, :, 1])
        width = croppedImage.shape[0]
        height = croppedImage.shape[1]
        croppedImage = np.reshape(croppedImage, (width * height, 3))
        for i in range(width * height):
            channels.append(croppedImage[i, :])

    return np.array(channels)


def detectBuoy(image, mean, sd, weights, clusters):
    data = np.reshape(image.shape[0]*image.shape[1], 3)
    Pc_x = np.zeros((len(data), clusters))
    combinedPc_x = np.zeros((len(data), clusters))

    for cluster in range(clusters):
        Pc_x[:, cluster:cluster + 1] = np.reshape(weights[cluster] * gaussian(data, mean[cluster], sd[cluster]),
                                                  (len(data), 1))
        combinedPc_x = Pc_x.sum(1)

    combinedPc_x = np.reshape(combinedPc_x, (image.shape[0], image.shape[1]))
    combinedPc_x[combinedPc_x > np.max(combinedPc_x) / 3.5] = 255
    output = np.zeros_like(image)
    output[:, :, 0] = combinedPc_x
    output[:, :, 1] = combinedPc_x
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
        else:
            cv2.imshow("Orange Buoy Detection", image)
    else:
        cv2.imshow("Orange Buoy Detection", image)
    return image


clusters = 2
images = []
parameters = [[9,7],[8.5, 9.5, 3.0]]
yellowData = accessTrainingData("Data/yellow")
print(len(yellowData))
orangeData = accessTrainingData("Data/orange")
greenData = accessTrainingData("Data/green")
# print(len(yellowData[1]))
meany, covy, weightsy = EM(yellowData, clusters)
meano, covo, weightso = EM(orangeData, clusters)
meang, covg, weightsg = EM(greenData, clusters)
mean = [meany, meano, meang]
cov = [covy, covo, covg]
weights = [weightsy, weightso, weightsg]
#
# Reading and Writing Video
cap = cv2.VideoCapture('detectbuoy.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detectbuoy3Dguass.avi', fourcc, 5.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = detectBuoy(frame, mean[0], cov[0], weights[0], clusters)
    frame = detectBuoy(frame, mean[1], cov[1], weights[1], clusters)
    frame = detectBuoy(frame, mean[2], cov[2], weights[2], clusters)
    images.append(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for image in images:
    out.write(image)
out.release()
cap.release()
cv2.destroyAllWindows()

