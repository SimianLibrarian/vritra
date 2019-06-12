from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
#import matplotlib.pyplot as plt
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
for c in cnts:
    shape=sd.detect(c)
    if shape!="unidentified":
        print(shape)
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        c=c.astype("float")
#        c=c[:]*ratio
        c=c.astype("int")
#        print([c])
        cv2.drawContours(resized, c, -1, (0,0, 200), 2)
        cv2.imshow("Image",resized)
#        cv2.imshow("Image",[c])


        break
cv2.waitKey(2000)