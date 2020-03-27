import cv2
import math


def saveCroppedImage(croppedImage):
    # cv2.imwrite("testData/green/green" + str(count) + "_test.jpg", croppedImage)
    # cv2.imwrite("testData/orange/orange" + str(count) + "_test.jpg", croppedImage)
    cv2.imwrite("testData/yellow/yellow" + str(count) + "_test.jpg", croppedImage)


def croppedImage(frame, center, radius):
    global count
    cropImage = frame[center[1] - radius: center[1] + radius, center[0] - radius: center[0] + radius]
    # cropImage = cv2.resize(cropImage, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("cropped", cropImage)
    count += 1
    saveCroppedImage(cropImage)
    # return cropImage


# event: Left button of the mouse pressed
# x: The x-coordinate of the event.
# y: The y-coordinate of the event.
# flag: Any relevant flags passed by OpenCV.
# params: Any extra parameters supplied by OpenCV.
def regionOfInterest(event, x, y, flag, params):
    global center, corner, croppingStatus
    if event == cv2.EVENT_LBUTTONDOWN:
        center = (int(x), int(y))
        croppingStatus = True
        # count += 1

    elif event == cv2.EVENT_LBUTTONUP:
        corner = (int(x), int(y))
        croppingStatus = False
        # count = count + 1
        radius = int(math.sqrt((corner[0] - center[0]) ** 2 + (corner[1] - center[1]) ** 2))
        croppedImage(frame, center, radius)


# Global Parameters
croppingStatus = False
center = (0, 0)
corner = (0, 0)
count = 0
######################################################
#              Reading Video 
#####################################################
name = "detectbuoy.avi"
cap = cv2.VideoCapture(name)

while cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape)
    cv2.imshow("name", frame)
    cv2.setMouseCallback("name", regionOfInterest)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
