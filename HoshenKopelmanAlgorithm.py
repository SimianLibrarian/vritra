import numpy as np
import matplotlib.pyplot as plt
import cv2,imutils
from pyimagesearch.shapedetector import ShapeDetector

def union(Label,i,j):
    label_left=Label[i-1,j]
    label_above=Label[i,j-1]
    Label[Label==label_left]=label_above
    return Label

def HK_label(M):
    N1,N2=len(M),len(M[0])
    Label=np.asarray([[0]*N1]*N2)
    largest_label=0
    for i in range(N1):
        for j in range(N2):
            if M[i][j]==1:
                if i==0 and j ==0:
                    largest_label=largest_label+1
                    Label[i,j]=largest_label
                else:
                    if M[i-1][j]==0 and M[i][j-1]==0:
                        largest_label=largest_label+1
                        Label[i,j]=largest_label
                    elif M[i-1][j]==1 and M[i][j-1]==0:
                        Label[i,j]=Label[i-1,j]
                    elif M[i-1][j]==0 and M[i][j-1]==1:
                        Label[i,j]=Label[i,j-1]
                    elif M[i-1][j]==1 and M[i][j-1]==1:
                        Label=union(Label,i,j) #merge left and above clusters
                        Label[i,j]=Label[i-1,j]   
            else:
                Label[i,j]=0
    for i in range(1,len(np.unique(Label))):#rearrange label values
        Label[Label==np.unique(Label)[i]]=i
    return Label
#%%
#M=cv2.imread()
sd=ShapeDetector()
img=cv2.imread("../PackingAnalysis/slice.tif") #it's the fish egg picture
img = imutils.resize(img, width=400) #image resized
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred=cv2.medianBlur(gray,5)
blurred=cv2.bilateralFilter(gray,15,15,15)
#blurred[blurred>np.mean(blurred)*1.4]=np.mean(blurred)
#    blurred=255-blurred
ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself

thresh[thresh==255]=1   
#thresh=1-thresh
Label=HK_label(thresh)
q=len(np.unique(Label))-1
#mcs=np.count_nonzero(Label)/q#mean cluster size
size=[np.count_nonzero((Label==np.unique(Label)[i]).astype(int)) for i in range(1,q)]
mcs=np.mean(size)
gen=[[0]*len(Label[0])]*len(Label)

for i in range(1,q):
    Label_copy=((Label==np.unique(Label)[i]).astype(int))
    np.asarray(Label_copy,dtype="uint8")
    print(np.count_nonzero(Label_copy))
#    if np.count_nonzero(Label_copy)<mcs and np.count_nonzero(Label_copy)>2:
    if np.count_nonzero(Label_copy)>=1:
#        plt.imshow(Label_copy),plt.show()
#        print(np.sum(Label_copy))
#        cat=np.array(Label_copy ,dtype=np.uint8)
#        cnts=cv2.findContours(cat,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        cnts=imutils.grab_contours(cnts)
#        if sd.detect(cnts[0])=='circle':
        gen = gen+ np.array(Label_copy ,dtype=np.uint8)

fig=plt.figure(figsize=(12,6))
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(thresh-gen),plt.show()
#        cnts=cv2.findContours(gen,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        cnts=imutils.grab_contours(cnts)
#        if len(cnts)==1:
#            if sd.detect(cnts)=="circle":
#                print("Inner egg!")
    
#    Label_copy=np.asarray([[0]*len(Label[0])]*len(Label))
#    for j in range(len(Label)):
#        for k in range(len(Label[0])):
#            if Label[j,k]==np.unique(Label)[i]:
#                Label_copy=1
            
    
#fig=plt.figure(figsize=(18,9))
#plt.subplot(131),plt.imshow(img)
#plt.subplot(132),plt.imshow(thresh)
#plt.subplot(133),plt.imshow(Label)
#print("Number of clusters : "+str(len(np.unique(Label))-1))
#print("Largest label : "+str(np.max(Label)))