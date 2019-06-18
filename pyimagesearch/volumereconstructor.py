import cv2,imutils,numpy as np, matplotlib.pyplot as plt, random as rng
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from os import listdir
import xml.etree.ElementTree as ET

class VolumeReconstructor:
    def __init__(self):
        pass
# The first step should be to identify the number of circles one can get from a picture
# 1- Get how many circles N are possible to identify : repetition over x slices
# 2- Declare N arrays, which will be incrementally concatenated
# 3- Compute each array/volume separately

    def counting(self,fname):
        labels=np.asarray(np.genfromtxt(fname,usecols=0),dtype=int)
        t=np.asarray(np.genfromtxt(fname,usecols=(1,2,3)))     
        with open(fname) as f:
            for line in f.readlines():
                li = line.lstrip()
                if li.startswith("#"):
                                 i1,i2=1,li.rfind(" ")
        l=float(''.join( [li[i1+j] for j in range(i2-i1) if li[i1+j].isdigit()==True] ))
        L=float(''.join([li[i2+j] for j in range(len(li)-i2) if li[i2+j].isdigit()==True]))
        t[:,0],t[:,1],t[:,2],N=t[:,0]/l,t[:,1]/L,t[:,2]/L,0
        indices=list()
        for k in range(len(t)):
            a=[i-1 for i in range(k+1,len(t)) if np.allclose(t[k][0:2],t[i-1][0:2],rtol=5e-2)==True]
            if len(a)>1 and np.any( [a[i] in indices[j] for j in range(len(indices)) for i in range(len(a))  ] )==False:
                indices.append(a)           
                N=N+1
#        ind_bis=indices
        for j in range(len(indices)):
            if j>=len(indices):
                break
            p=labels[indices[j]]
            if np.all(np.linspace(min(p),max(p),num=len(p))==sorted(p))!=True:
                try:
                    del indices[j]
                    j=j-1
                except:
                    print("Error")
#        indices=ind_bis
        return N,indices
                #np.any( [[0,2][i] in indices[j] for j in range(len(indices)) for i in range(2)  ] )
#        q=np.concatenate([[labels[0]],t[0]])
#        for i in range(len(a)):
#            q0=np.concatenate([[labels[a[i]]],t[a[i]]])
#            q=[q,q0]
    def expanding(self,fname):
        N,indices=counting(self,fname)
        
        
    def slicing(self,fname):
        #get the infos from the file
        #if 2 circles in 2 =/= slices have center different from less than x %, the 2 circles are bound together
        labels=np.asarray(np.genfromtxt(fname,usecols=0))
        t=np.asarray(np.genfromtxt(fname,usecols=(1,2)))
        #first step is to rescale all dimensions so as to better understand the position variations
        with open(fname) as f:
            for line in f.readlines():
                li = line.lstrip()
                if li.startswith("#"):
                                 i1,i2=1,li.rfind(" ")
        l=float(''.join( [li[i1+j] for j in range(i2-i1) if li[i1+j].isdigit()==True] ))
        L=float(''.join([li[i2+j] for j in range(len(li)-i2) if li[i2+j].isdigit()==True]))
        t[:,0],t[:,1]=t[:,0]/l,t[:,1]/L
        for i in range(len(np.unique(labels))):
            args2=[j for j in range(len(labels)) if labels[j]==np.unique(labels)[i]] 
            args1=[j for j in range(len(labels)) if abs(labels[j]-np.unique(labels)[i])==1]
#            if (abs(t[args2]-t[args1])<=0.01)==[True,True]:
#                print("Yeah")
        temp=[t[args2][i] for i in range(len(t[args2])) if np.all(abs(t[args2][i]-t[args1])<=0.01)]
        r=t[args1]
        if len(temp)!=0:
            r=np.concatenate([r,temp])
        return r
    
    def detection_verbose(self,fname):
        if type(fname)==str:
            img=cv2.imread(fname) #image importation
            resized = imutils.resize(img, width=800) 
        else:
            img=fname
            resized=img
            resized[resized<=60]=np.max(resized)
#            plt.imshow(resized)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color to gray conversion
        blurred=cv2.medianBlur(gray,5) #filtering the image to get better shape detection
#        blurred=cv2.GaussianBlur(gray, (5,5), 0)
        ret, thresh = cv2.threshold(blurred,200,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=15,labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]   
        labels = watershed(-D, markers, mask=thresh)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
#        rs=resized.copy()
        for label in np.unique(labels):
            if label == 0:
                continue 
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            if np.sum(mask)>100*255:
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)   
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
#                if (cX-len(resized)*0.5)**2+(cY-len(resized[0])*0.5)**2<=0.25*(len(resized)-10)**2:
                if (cX-len(resized)*0.5)**2+(cY-len(resized[0])*0.5)**2<=(len(resized)/2*0.95)**2:
                    cv2.drawContours(resized, c, -1, (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256)), 2)
                    cv2.circle(resized,(cX,cY),3, (255,0,0), -1)
            else:
                labels[labels==label]=0
        fig=plt.figure(figsize=(28,18))
        plt.subplot(131),plt.imshow(img),plt.axis('off')
        plt.subplot(132),plt.imshow(labels,cmap='plasma'),plt.axis('off')
        plt.subplot(133),plt.imshow(resized),plt.axis('off')
        plt.show()
    
    def detection_silent(self,fname,ofname,mode='a'):
        if type(fname)==str:
            img=cv2.imread(fname) #image importation
            resized = imutils.resize(img, width=800) 
        else:
            img=fname
            resized=img
            resized[resized<=60]=np.max(resized)
        resized = imutils.resize(img, width=800) 
#        resized=img
        ratio = img.shape[0] / float(resized.shape[0]) #compression ratio
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #color to gray conversion
        blurred=cv2.medianBlur(gray,5) #filtering the image to get better shape detection
#        blurred=cv2.GaussianBlur(gray, (5,5), 0)
        ret, thresh = cv2.threshold(blurred,10,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #binarisation ; Otsu's method chooses threshold itself
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=15,labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]   
        labels = watershed(-D, markers, mask=thresh)
        rs=resized.copy()
        f=open(ofname,mode)
        for label in np.unique(labels):
            if label == 0:
                continue 
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            if np.sum(mask)>100*255:
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)   
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                cv2.drawContours(rs, c, -1, (rng.randint(0,256),rng.randint(0,256),rng.randint(0,256)), 2)
                cv2.circle(rs,(cX,cY),3, (255,0,0), -1)
                f.write(str(float(cX)/len(rs))+' '+str(float(cY)/len(rs[0]))+'\n')
            else:
                labels[labels==label]=0
        f.close()

    def crop(self,folder):
        fname=folder+listdir(folder)[-1]
        im = cv2.imread(fname)
        resized = im
        ratio = im.shape[0] / float(resized.shape[0])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh=255-thresh
        cnts=cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        hcy=cnts[-1]
        c=imutils.grab_contours(cnts)
        for i in range(len(c)):
            if hcy[0][i][2]==-1 and cv2.arcLength(c[i],True)>=200:
                    x,y,w,h = cv2.boundingRect(c[i])
#                    cv2.drawContours(resized, c[i], -1, (0,100*i, 200), 2)
        return x,y,w,h
#        rs=gray[y+5:y+h-5,x+5:x+w-5]
#        cv2.imshow("Image",rs),cv2.waitKey(),cv2.destroyAllWindows()

    def xml_encoder(Img_Num,Label,cX,cY):
        data = ET.Element('data') 
        particles = ET.SubElement(data, 'particles') 
        
        img_num = ET.SubElement(particles,'img_num')
        img_num.set('name','Image number')
        img_num.text = str(Img_Num)
        
        label = ET.SubElement(particles,'label')
        label.set('name','Label')
        label.text = str(Label)
        
        center = ET.SubElement(particles, 'center')  
        center.set('name','Center Position')
        
        cx = ET.SubElement(center,'cX')
        cx.set('name','cX')
        cx.text = str(cX)
        
        cy = ET.SubElement(center,'cY')
        cy.set('name','cY')
        cy.text = str(cY)
#        item2 = ET.SubElement(items, 'item')  
#        item1.set('name','item1')  
#        item2.set('name','item2')  
#        item1.text = 'item1abc'  
#        item2.text = 'item2abc'
        mydata = ET.tostring(data)  
        myfile = open("CenterPositions.xml", "w")  
        myfile.write(mydata) 
        myfile.close()

# xml_encoder("10",1.0,2.0)




















