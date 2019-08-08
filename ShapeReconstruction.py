import cv2,os,subprocess,imutils,numpy as np, matplotlib.pyplot as plt, random as rng
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from pyimagesearch.shapedetector import ShapeDetector
#import numpy as np
img_folder="./SliceY Droplets -10um/"
content = os.listdir(img_folder)
content=[content[i] for i in range(len(content)) if ".tif" in content[i]]
f=open("CirclesPositions.txt","w")
for j in range(len(content)):
    p=subprocess.Popen("python3.7 detect_shapes.py -i "+img_folder+str(content[j])+" -o CirclesPositions.txt",shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        f.write(str(line))
f.close()


#This part is dedicated to detect droplets in the case of Giustiniani's mixture

#Import image into 'img' > resize in in 'resized', get old/new size in 'ratio'
# > convert to gray in 'gray' > apply median filter in 'blurred'
# > Otsu's binarisation in 'thresh' > fill tiny black dots in 'opening'
# > transform in gray map in distance from border in 'dist', then normalise it 
# > threshold dist in 'dist'

#%%
#import shapedetector as sd

def is_inside(thresh,center,radius):
    for i in range(len(thresh)):
        for j in range(len(thresh[i])):
            if (i-int(center[0]))**2+(j-int(center[1]))**2>(radius**2):
                thresh[i,j]=0
    return thresh
#plt.imshow(is_inside(thresh,[x,y],radius))
def segmentation(fname,show="no",write="no",fout="ContourShape.txt"):
    img=cv2.imread(fname)
    img = imutils.resize(img, width=1100) #image resized
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    blurred=cv2.medianBlur(gray,5)
    blurred=cv2.bilateralFilter(gray,15,15,15)
    blurred[blurred>np.mean(blurred)*1.4]=np.mean(blurred)
#    blurred=255-blurred
    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself
#    plt.imshow(thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contours finding for the dist shapes
    cnts = imutils.grab_contours(cnts)
    imax=np.argmax([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
    (x,y),radius = cv2.minEnclosingCircle(cnts[imax+1])
    blurred=cv2.bilateralFilter(gray,15,15,15)
    blurred[blurred<np.mean(blurred)*0.725]=np.mean(blurred)*1.1
    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself

    thresh=is_inside(thresh,[x,y],radius)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours finding for the dist shapes
    cnts_copy=cnts
    cnts = imutils.grab_contours(cnts)
    if show=="yes" or show=="Yes":
        for c in cnts:
            if cv2.contourArea(c)>200:
                cv2.drawContours(gray,c,-1,(0,200,200),1)
                cv2.drawContours(img,c,-1,(randint(0,255),randint(0,255),randint(0,255)),2)
                cv2.imshow("detect",img)
    #            cv2.waitKey(100)
        cv2.waitKey()
        cv2.destroyAllWindows()
    if write=="yes" or write=="Yes":
        contour_id(imutils.grab_contours(cnts_copy),fout=fout)
    return cnts_copy

def contour_id(cnts,fout="ContourShape.txt"): #take contours and write their shape in file if circle enough
    sd=ShapeDetector()
    i=0
    f=open(fout,"w")
    for c in cnts:
        if sd.detect(c)=="circle":
            i=i+1
            for j in range(len(c)):
                f.write(str(i)+" "+str(c[j,0,0])+" "+str(c[j,0,1])+"\n")
    f.close()

    

fname=img_folder+"slice00058.tif"
cnts=segmentation(fname,write="yes")
#contour_id(imutils.grab_contours(cnts))
#%%


img=cv2.imread(img_folder+"slice00058.tif") #image importation
resized = imutils.resize(img, width=1300) #image resized
ratio = img.shape[0] / float(resized.shape[0]) #compression ratio
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #color to gray conversion
blurred=cv2.medianBlur(gray,3) #filtering the image to get better shape detection
blurred[blurred<np.mean(blurred)*0.75]=np.mean(blurred)*1.15
ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself

plt.imshow(thresh)
kernel = np.ones((3,3),np.uint8) #?
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)#thos pixels are the ones that we're sure are from the background
##distanceTransform associe a chaque pixel un niveau de gris proportionnel a la distance qui le separe du bord
dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
ret, dist = cv2.threshold(dist, 0.09, 1.0, cv2.THRESH_BINARY) #thresholding the distance map to recompose the shapes
#ret, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY) #thresholding the distance map to recompose the shapes
kernel1 = np.ones((2,2), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1) #expanding a little the shapes so as to remove inner black dots
##cv2.imshow('Peaks', dist),cv2.waitKey(),cv2.destroyAllWindows()
#
dist_8u = dist.astype('uint8')
_, contours = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(dist.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contours finding for the dist shapes
cnts = imutils.grab_contours(cnts)
a,b = cv2.findContours(dist.astype("uint8") ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
##cv2.startWindowThread()
##cv2.namedWindow("dist")
#for c in cnts:
#    cv2.drawContours(resized,c,-1,(0,0,200),2)
#    cv2.imshow("dist",resized)
#    cv2.waitKey(2)
#cv2.waitKey()
#cv2.destroyAllWindows()
##plt.imshow(markers,cmap='Greys'),plt.show()
#
##      cv2.drawContours(dist,contours[10],-1,(0,255,0),2),cv2.imshow("dist",resized),cv2.waitKey(),cv2.destroyAllWindows()
##markers = np.zeros(dist.shape, dtype=np.int32)
##for i in range(len(cnts)):
##    cv2.drawContours(markers, cnts, i, (i+1), -1)

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=10,	labels=thresh)
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]   
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
#localMax = peak_local_max(D, indices=False, min_distance=20,
#	labels=thresh)   
#cv2.circle(markers, (5,5), 3, (255,255,255), -1)

for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
    if label == 0:
        continue 
	# otherwise, allocate memory for the label region and draw
	# it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    if np.sum(mask)>60*255:
#    print(np.sum(mask==255))
	# detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)   
        c = max(cnts, key=cv2.contourArea)
    #    ((x, y), r) = cv2.minEnclosingCircle(c)
    #    print(cv2.contourArea(c))
    #    if np.sum(mask==255)>=100:
        cv2.drawContours(resized, cnts, -1,(0,0, 200), 1)
#    cv2.waitKey(5)
cv2.imshow("Image",resized),cv2.waitKey(),cv2.destroyAllWindows()
#    else:
#        print("too smol")
	# draw a circle enclosing the object
#	cv2.circle(gray, (int(x), int(y)), int(r), (0, 255, 0), 2) 
# show the output image
#cv2.imshow("Output",resized),cv2.waitKey(),cv2.destroyAllWindows()
#%%

#a=cv2.watershed(dist_8u,markers)
#plt.imshow(a),plt.show()
## the watershed alogrithm expands the tiles too much and covers the whole surface
##How does it work ? How can we limit it so it separates correctly the droplets from the background ?
##cv2.watershed(resized, markers)
#mark = markers.astype('uint8')
#mark = cv2.bitwise_not(mark)
##cv2.waitKey(),cv2.destroyAllWindows()
#colors = []
#for contour in cnts:
#    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
#
#
#dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
#for i in range(markers.shape[0]):
#    for j in range(markers.shape[1]):
#        index = markers[i,j]
#        if index > 0 and index <= len(cnts):
#            dst[i,j,:] = colors[index-1]
#cv2.imshow('Final Result', dst),cv2.waitKey(),cv2.destroyAllWindows()
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
ret, sure_fg = cv2.threshold(dist,0.7*dist.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#ret, markers = cv2.connectedComponents(sure_fg)
#markers = markers+1
#markers[unknown==255] = 0
#markers = cv2.watershed(img,markers)
#img[markers == -1] = [255,0,0]
resized=255-resized
sure_bg=255-sure_bg
sure_fg=255-sure_fg


plt.figure(figsize=(16,6))
plt.subplot(131)
plt.imshow(resized,cmap='Greys')
plt.subplot(132)
plt.imshow(sure_bg,cmap="Greys")
plt.subplot(133)
plt.imshow(sure_fg,cmap="Greys")

plt.figure(figsize=(12,6))
plt.imshow(dist)

    
