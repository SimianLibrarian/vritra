from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
#import matplotlib.pyplot as plt
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
ap.add_argument("-t","--time",required=False)
ap.add_argument("-o","--output",required=True)
args = vars(ap.parse_args())

#outfile=args["output"]
f=open(args["output"],"a")
image=cv2.imread(args["image"])
resized = imutils.resize(image, width=800)
ratio = image.shape[0] / float(resized.shape[0])
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (4,4), 0)
#blurred=cv2.blur(gray,(5,5))
blurred=cv2.medianBlur(gray,7)
#blurred=gray
#thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)[1]
ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh=255-thresh
cv2.imshow("thresh",thresh),cv2.waitKey(2000)

#thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]
f.write("# "+str(len(resized))+" "+str(len(resized[0]))+"\n")
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
for c in cnts:
    shape=sd.detect(c)
    if shape!="unidentified":
#        print(shape)
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        c=c.astype("float")
        c=c.astype("int")
        (x,y),radius=cv2.minEnclosingCircle(c)
#        radius=int(radius)
#        print(args["image"])
        label=''.join([(s) for s in args["image"] if s.isdigit()]) 
        f.write(str(label)+" "+str(x)+" "+str(y)+" "+str(radius)+"\n")
        cv2.drawContours(resized, c, -1, (0,0, 200), 2)
        cv2.imshow("Image",resized)
f.close()
if args["time"]:
    cv2.waitKey(int(args["time"]))
else:
    cv2.waitKey(2000)