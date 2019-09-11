#from pyfiltering.imagefiltering import *
import pyfiltering.imagefiltering as imf
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os 
import scipy as scp
import skimage as sk
import skimage.morphology, skimage.segmentation, skimage.measure #don't know why, but need to be imported this way

i_folder="./DatasICS/SliceYDroplets10um/SlicesY/"
dat=imf.StackImportation(i_folder,4)
dat=imf.PictureNormalisation(dat)
intensity_kernel,spatial_kernel=0.1,4
bilateral=imf.BilateralFiltering(dat,intensity_kernel,spatial_kernel)
bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
cropped=imf.StackCropping(bilateral,0.4)[0]
rw=imf.StackSegmentation(bilateral,threshold=0.4,imin=0.4,imax=0.7)
#rw=sk.morphology.opening(rw,sk.morphology.cube(10))
rw=sk.morphology.remove_small_objects(rw==2,min_size=200)
plt.figure(),plt.imshow(cropped[0],cmap='gray'),plt.contour(rw[0],[1]),plt.show()
labels= sk.measure.label(rw,background=0)
labels = (labels + 1).astype(np.uint8)
                ##%%this part is to do the beam hardening correction prior to segmentation. Is is required, 
                ##  or is the beam correction already applied ? Ask Damien !
                #i_folder="./DatasICS/SliceYDroplets10um/SlicesY/"
                ##get mask > correct beam hardening > filter,crop around mask, threshold, segment
                ##1:Get bounding box and mask
                #dat=imf.StackImportation(i_folder,2)
                #dat_copy=np.copy(dat)
                #dat_copy,intensity_kernel,spatial_kernel=imf.PictureNormalisation(dat_copy),0.1,4
                #bilateral=imf.BilateralFiltering(dat_copy,intensity_kernel=intensity_kernel,
                #                                 spatial_kernel=spatial_kernel)
                #bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
                #crop,bbox,mask=imf.StackCropping(bilateral,0.4)
                ##Correct Beam Hardening
                #dat=imf.PictureNormalisation(dat,0.,100.)
                #dat=imf.StackRadialCorrection(dat)
                #dat=imf.PictureNormalisation(dat,0,100)
                ##hi_bilateral=sk.exposure.histogram(bilateral)
                ##Filter
                ##dat=imf.PictureNormalisation(dat)
                #bilateral=imf.BilateralFiltering(dat,intensity_kernel,spatial_kernel)
                #bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
                ##bilateral=bilateral[bbox[0]]
                ##!!!!!
                ##I need to find a way to erode the mask
                #mask=sk.morphology.erosion(mask,sk.morphology.cube(20))
                ##I need to find a way to erode the mask
                ##!!!!!
                #bilateral=bilateral[bbox[0]]*mask
                #rw=imf.StackSegmentation(bilateral,cropping="no",mask=mask,threshold=0.2,imin=0.3825,imax=0.53)
                #rw=sk.morphology.opening(rw,sk.morphology.cube(10))
                #plt.figure(),plt.imshow(cropped[0],cmap='gray'),plt.contour(rw[0],[1]),plt.show()


#dat=imf.StackImportation(i_folder,4)
#dat=imf.PictureNormalisation(dat)
#intensity_kernel,spatial_kernel=0.1,4
#bilateral=imf.BilateralFiltering(dat,intensity_kernel=intensity_kernel,
#                                 spatial_kernel=spatial_kernel)
#bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
#mask=imf.StackCropping(bilateral,0.4)[1]
#dat=imf.StackRadialCorrection(cropped)
#dat=imf.PictureNormalisation(dat)
#intensity_kernel,spatial_kernel=0.1,4
#bilateral=imf.BilateralFiltering(dat,intensity_kernel,spatial_kernel)
#bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
#cropped=imf.StackCropping(bilateral,0.4)[0]
#rw=imf.StackSegmentation(bilateral,threshold=0.4,imin=0.4,imax=0.7)
#plt.figure(),plt.imshow(cropped[0],cmap='gray'),plt.contour(rw[0],[1]),plt.show()

#%%
def filter_compare(dat,bilateral): #func to compare raw/filtered image
    hi_dat=sk.exposure.histogram(dat)
    hi_bilateral=sk.exposure.histogram(bilateral)
    fig=plt.figure(figsize=(12,12))
    plt.subplot(221),plt.imshow(dat[0])
    plt.subplot(222),plt.imshow(bilateral[0])
    plt.subplot(212),plt.plot(hi_dat[1],hi_dat[0],label="img"),
    plt.plot(hi_bilateral[1],hi_bilateral[0],label="bilateral")
    plt.legend(),plt.figure()