import os
import subprocess
#import numpy as np
content = os.listdir("./Giustiniani/")
content=[content[i] for i in range(len(content)) if ".tif" in content[i]]
f=open("CirclesPositions.txt","w")
for j in range(len(content)):
    p=subprocess.Popen("python3.7 detect_shapes.py -i ./Giustiniani/"+str(content[j])+" -o CirclesPositions.txt",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    print(stdout)
    for line in p.stdout.readlines():
        f.write(str(line))
#    retval = p.wait()
f.close()

#%% Second part : recognize the circles appearing on neighboring slices
from pyimagesearch.volumereconstructor import VolumeReconstructor
fname="CirclesPositions.txt"
vr=VolumeReconstructor()
N,indices=vr.counting(fname)
labels=np.asarray(np.genfromtxt(fname,usecols=0,dtype=int))
#t=np.asarray(np.genfromtxt(fname,usecols=(1,2,3))) 
t=np.asarray(np.genfromtxt(fname,usecols=(1,2,3)))
content=os.listdir("./")
content= [content[i] for i in range(len(content)) if content[i][-4::]==".tif"]
#np.any([str(labels[0]) in content[i] for i in range(len(content))])
#taking care of the 1st sphere
pmin,pmax=min(labels[indices[0]]),max(labels[indices[0]])
if np.any([str(pmax+1) in content[i] for i in range(len(content))])==True:
    for k in range(len(content)):
        if str(pmax+1) in content[k]:
            fname=content[k]
            break
#    f=open(fname,"a")
    image=cv2.imread(fname)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]
    r0=inscribed_circle(thresh,t[indices[0]][0][0:2] )

#    cv2.circle(resized, [int(t[indices[0]][0][0]),int(t[indices[0]][0][1])],int(r0),(0,200,0),-1)
    cv2.circle(resized, (121,260), 1, (0, 255, 0), -1)
    cv2.imshow("Image",resized)
    cv2.waitKey(0)
    #means we can expand it with pics with higher labels
# thresh[int(t[indices[0],0:2][0][1]),int(t[indices[0],0:2][0][0])]
    
def inscribed_circle(thresh,pos):
    thresh=np.asarray([map(lambda x: True if x==255 else False,thresh[i]) for i in range(len(thresh))])
    if thresh[int(pos[0]),int(pos[1])]==True:
        print("It's a good point")
    r=10 #the first test circle. will check if every pixel in a radius r from (pos[0],pos[1]) is white
    R=np.linspace(1,r,num=r)
#    [thresh[int(pos[0]-R)]]
    a=[thresh[int(pos[0]-R[i]):int(pos[0]+R[i]),int(pos[1]-(r*r-R[i]*R[i])**.5):int(pos[1]+(r*r+R[i]*R[i])**.5)] for i in range(len(R))]
#    m=[int(False in a[i]) for i in range(len(a))]
    m=np.argmax([False in a[i] for i in range(len(a))])-1
    return R[m]
#%%
#   import cv2,imutils,numpy as np, matplotlib.pyplot as plt, random as rng
#This part is dedicated to detect droplets in the case of Giustiniani's mixture
img=cv2.imread("./Giustiniani/img_1909.tif") #image importation
resized = imutils.resize(img, width=800) #image resized
ratio = img.shape[0] / float(resized.shape[0]) #compression ratio
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #color to gray conversion
blurred=cv2.medianBlur(gray,5) #filtering the image to get better shape detection
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself

kernel = np.ones((3,3),np.uint8) #?
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#sure_bg = cv2.dilate(opening,kernel,iterations=3)#thos pixels are the ones that we're sure are from the background
#distanceTransform associe à chaque pixel un niveau de gris proportionnel à la distance qui le sépare du bord
dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('Distance Transform Image', dist)
cv2.waitKey(),cv2.destroyAllWindows()

ret, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY) #thresholding the distance map to recompose the shapes
cv2.imshow('Distance Transform Image', dist)
cv2.waitKey(),cv2.destroyAllWindows()


kernel1 = np.ones((2,2), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1) #expanding a little the shapes so as to remove inner black dots
cv2.imshow('Peaks', dist),cv2.waitKey(),cv2.destroyAllWindows()

dist_8u = dist.astype('uint8')
#_, contours = cv2.findContoursl(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(dist.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours finding for the dist shapes
cnts = imutils.grab_contours(cnts)
#a,b = cv2.findContours(dist.astype("uint8") ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#cv2.startWindowThread()
#cv2.namedWindow("dist")
for c in cnts:
    cv2.drawContours(resized,c,-1,(0,0,200),2)
    cv2.imshow("dist",resized)
    cv2.waitKey(5)
cv2.waitKey()
cv2.destroyAllWindows()


#      cv2.drawContours(dist,contours[10],-1,(0,255,0),2),cv2.imshow("dist",resized),cv2.waitKey(),cv2.destroyAllWindows()
markers = np.zeros(dist.shape, dtype=np.int32)

for i in range(len(cnts)):
    cv2.drawContours(markers, cnts, i, (i+1), -1)
#cv2.circle(markers, (5,5), 3, (255,255,255), -1)
plt.imshow(markers,cmap='Greys'),plt.show()

# the watershed alogrithm expands the tiles too much and covers the whole surface
#How does it work ? How can we limit it so it separates correctly the droplets from the background ?
cv2.watershed(resized, markers)
mark = markers.astype('uint8')
mark = cv2.bitwise_not(mark)
#cv2.waitKey(),cv2.destroyAllWindows()
colors = []
for contour in cnts:
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))


dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(cnts):
            dst[i,j,:] = colors[index-1]
cv2.imshow('Final Result', dst),cv2.waitKey(),cv2.destroyAllWindows()
# from pyimagesearch.shapedetector import ShapeDetector
#sd=ShapeDetector()
#for c in cnts:
#    shape=sd.detect(c)
#    M = cv2.moments(c)
#    cX = int((M["m10"] / M["m00"]) * ratio)
#    cY = int((M["m01"] / M["m00"]) * ratio)
#    print(cX,cY)

#
#
##cv2.imshow('Markers', markers*10000),cv2.waitKey(),cv2.destroyAllWindows()
#
#
#ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)
##ret, markers = cv2.connectedComponents(sure_fg)
##markers = markers+1
##markers[unknown==255] = 0
##markers = cv2.watershed(img,markers)
##img[markers == -1] = [255,0,0]
#resized=255-resized
#sure_bg=255-sure_bg
#sure_fg=255-sure_fg
#plt.figure(figsize=(16,6))
#plt.subplot(131)
#plt.imshow(resized,cmap='Greys')
#plt.subplot(132)
#plt.imshow(sure_bg,cmap="Greys")
#plt.subplot(133)
#plt.imshow(sure_fg,cmap="Greys")
#
#plt.figure(figsize=(12,6))
#plt.imshow(dist_transform)

    
