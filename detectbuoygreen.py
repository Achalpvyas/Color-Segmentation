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
        croppedImage = croppedImage[:, :, 1]
        for i in range(croppedImage.shape[0]):
            channels.append(croppedImage[i, :])

    return np.concatenate(channels)


def detectBuoy(image, mean, sd, weights, clusters):
    images = []
    data = image[:, :, 1].ravel()

    Pc_x = np.zeros((len(data), clusters))
    combinedPc_x = np.zeros(len(data))

    for cluster in range(clusters):
        Pc_x[:, cluster:cluster + 1] = np.reshape(weights[cluster] * gaussian(data, mean[cluster], sd[cluster]),
                                                  (len(data), 1))

    combinedPc_x = np.reshape(Pc_x.sum(1), len(data))

    combinedPc_x = np.reshape(combinedPc_x, (image.shape[0], image.shape[1]))
    combinedPc_x[combinedPc_x > np.max(combinedPc_x) / 1.8] = 255
    output = np.zeros_like(image)
    output[:, :, 1] = combinedPc_x
    blur = cv2.medianBlur(output, 3)
    cv2.imshow('Median Blur', blur)
    edged = cv2.Canny(blur, 40, 150)
    cont, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    (cnts_sorted, boundingBoxes) = contours.sort_contours(cont, method="right-to-left")
    hull = cv2.convexHull(cnts_sorted[0])
    (x, y), radius = cv2.minEnclosingCircle(hull)
    if radius > 3:
        cv2.circle(image, (int(x), int(y) - 1), int(radius + 1), (0, 255, 0), 4)
        cv2.imshow("Detecting Green buoy", image)
        images.append(image)
    else:
        cv2.imshow("Detecting Green buoy", image)
        images.append(image)
    return images


clusters = 3
greenData = accessTrainingData("Data/green")
# print(len(greenData))
mean, sd, weights = EM(greenData, clusters)

# Reading and writing  Video
cap = cv2.VideoCapture('detectbuoy.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detectbuoygreen.avi', fourcc, 5.0, (640, 480))
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
