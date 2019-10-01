import numpy as np
import scipy as scp
import scipy.stats

import os
from PIL import Image
import skimage as sk
import skimage.restoration,skimage.morphology




def OrganiseSlices(folder):
    content=os.listdir(folder)
    if np.sum(["slice" in content])>=1:
       content=[x for x in content if "slice" in x]
    if np.sum(["img" in content])>=1:
       content=[x for x in content if "img" in x]
    content_copy=content
    content=[[s for s in content[i]if s.isdigit()] for i in range(len(content))]
    content=[int("".join(content[i])) for i in range(len(content))]
    argsort=np.argsort(content)
    content_copy[argsort[0]]
    content=[content_copy[argsort[i]] for i in range(len(argsort))]
    return content
def PictureNormalisation(dat,m=0.5,M=99.5):
    vmin,vmax=scp.stats.scoreatpercentile(dat,(m,M))
    dat = np.clip(dat, vmin, vmax)
    dat = (dat - vmin) / (vmax - vmin)    
    return dat

def StackImportation(folder,N,step=1,skip=0):
    content=OrganiseSlices(folder)
    dat=[None]*N
    for i in range(N):
        im_frame = Image.open(folder+content[int(skip)+i*int(step)])
        dat[i] = np.array(im_frame.getdata()).reshape(im_frame.size[1::-1])
    return np.asarray(dat)
def BilateralFiltering(dat,intensity_kernel=0.1,spatial_kernel=4):
#    t1 = time()
#    bilateral=[None]*len(dat)
#    for i in range(len(dat)):
#        bilateral= restoration.denoise_bilateral(dat[i],multichannel=False)
    bilateral=sk.restoration.denoise_bilateral(dat,sigma_color=intensity_kernel,sigma_spatial=spatial_kernel)
    bilateral=np.asarray(bilateral)
    return dat    

def StackSave(data,i_folder,o_folder):
    content=OrganiseSlices(i_folder)
    for i in range(len(data)):
        result = Image.fromarray(data[i])
        result.save(o_folder+'bilateral_'+str(content[i]))


    
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def StackRadialProfile(dat,center):
    r=[None]*len(dat)
    for i in range(len(dat)):
        r[i]=radial_profile(dat[i],center)
    r=np.asarray(r)
    
    return [np.mean(r[:,i]) for i in range(len(r[0]))]

def RadialCorrection(img,data,center,r_profil):
    
    y,x=np.indices((img.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
#    plt.imshow(r),plt.show()
#    plt.plot(r_profil),plt.show()
    for i in range(len(r)):
        for j in range(len(r[i])):
            if type(r[i,j])!=np.int64:
                print(r[i,j])
            else:
                img[i,j]=img[i,j]/r_profil[r[i,j]]
    return img
def StackRadialCorrection(data):
    center=[len(data[0])/2,len(data[0][0])/2]
    r_profil=StackRadialProfile(data,center)
    for i in range(len(data)):
        RadialCorrection(data[i],data,[len(data[i])/2,len(data[i][0])/2],r_profil)
    return data
        
def StackCropping(img,threshold):
    sample=img>threshold
    sample=scp.ndimage.binary_fill_holes(sample)
    #sample=sk.morphology.opening(sample,sk.morphology.ball(2))
    sample=sk.morphology.opening(sample,sk.morphology.cube(12))
#    plt.imshow(sample[0])
    sample=sk.morphology.closing(sample,sk.morphology.cube(4))
    bbox = scp.ndimage.find_objects(sample)
    mask = sample[bbox[0]]
#    plt.imshow(mask[0])
    img=img[bbox[0]]    
    return img,bbox,mask

def StackSegmentation(img,cropping='yes',mask='no',threshold=0.4,imin=0.4,imax=0.7):
    if cropping=='yes':
        if mask=="no":
            img,bbox,mask=StackCropping(img,threshold)
        else:
            img,bbox=StackCropping(img,threshold)[0:2]
    else:
        if mask=="no":
            mask=StackCropping(img,threshold)[2]
#        else:
#            bbox=StackCropping(img,threshold)[1]        
    bi_copy=np.copy(img)
    bi_copy[bi_copy>imax]=imin/200
    bi_copy= bi_copy>imin
    bi_copy=scp.ndimage.binary_fill_holes(bi_copy)
    bi_copy=sk.morphology.binary_opening(bi_copy,sk.morphology.cube(2))
    bi_copy=bi_copy*mask[0]
    bi_copy=scp.ndimage.binary_fill_holes(bi_copy)
    bi_copy=sk.morphology.opening(bi_copy,sk.morphology.cube(4))
    bi_copy=sk.morphology.closing(bi_copy,sk.morphology.cube(4))
#    bi_copy=sk.morphology.remove_small_objects(bi_copy,min_size=20)
    binarised=np.copy(bi_copy)
    binarised=binarised+1
    rw=sk.segmentation.random_walker(img,binarised,beta=10,mode='cg')
    return rw

def ContourFinding_slice(l,t,fname="./DatasICS/SliceYDroplets10um/Contours.txt",mode="a"):
    with open(fname,mode) as o_file:
        o_file.write("#New slice "+str(t)+"\n")
        for i in np.unique(l):
            if i!=1:
                cnts = sk.measure.find_contours(l==i,level=0.99)   
                if len(cnts)==1:
                    for j in range(len(cnts[0])):
                        o_file.write(str(i)+" "+str(t)+" "+
                        str(int(cnts[0][j][0]))+" "
                        +str(int(cnts[0][j][1]))+"\n")              
        o_file.close()

def ContourFinding(labels,fname="./DatasICS/SliceYDroplets10um/Contours.txt"):
    for i in range(len(labels)):
        if i==0:
            ContourFinding_slice(labels[i],i,fname=fname,mode='w')
        else:
            ContourFinding_slice(labels[i],i,fname=fname,mode='a')

def Labels_SingleSlice(l,t,fname="./DatasICS/SliceYDroplets10um/Volumes.txt",mode="a"):       
    with open(fname,'w') as o_file:
        o_file.write("#Label x y z \n")
        o_file.write("# New slice "+str(t)+"\n")
        for i in range(len(l)):
            for j in range(len(l[i])):
                if l[i][j]!=1:
#                    o_file.write(str(l[i][j])+" "+str(i)+" "+str(j)+"\n")
                    o_file.write(str(l[i][j])+" "+str(i)+" "+str(j)+" "+str(t)+"\n")
    o_file.close()

def Labels(labels,fname="./DatasICS/SliceYDroplets10um/Volumes.txt"):
    for i in range(len(labels)):
        if i==0:
            Labels_SingleSlice(labels[i],i,fname=fname,mode="w")
        else:
            Labels_SingleSlice(labels[i],i,fname=fname,mode="a")