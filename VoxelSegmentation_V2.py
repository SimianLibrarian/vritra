#from pyfiltering.imagefiltering import *
import pyfiltering.imagefiltering as imf
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os 
import scipy as scp
import skimage as sk
import skimage.morphology, skimage.segmentation #don't know why, but need to be imported this way

i_folder="./DatasICS/SliceYDroplets10um/SlicesY/"
dat=imf.StackImportation(i_folder,4)
dat=imf.PictureNormalisation(dat)
intensity_kernel,spatial_kernel=0.1,4
bilateral=sk.restoration.denoise_bilateral(dat,sigma_color=intensity_kernel,sigma_spatial=spatial_kernel)
#bilateral=sk.restoration.denoise_bilateral(dat)
bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
#bilateral=imf.PictureNormalisation(bilateral)

sample=bilateral>0.42
sample=scp.ndimage.binary_fill_holes(sample)
#sample=sk.morphology.opening(sample,sk.morphology.ball(2))
sample=sk.morphology.opening(sample,sk.morphology.cube(12))
plt.imshow(sample[0])
sample=sk.morphology.closing(sample,sk.morphology.cube(4))
bbox = scp.ndimage.find_objects(sample)
mask = sample[bbox[0]]
plt.imshow(mask[0])
bilateral=bilateral[bbox[0]]


hi_bilateral=sk.exposure.histogram(bilateral)
#plt.plot(hi_bilateral[1],hi_bilateral[0])
bi_copy=np.copy(bilateral)
#imin,imax=scp.stats.scoreatpercentile(bi_copy,(15,60))
imin,imax=0.4,0.70
bi_copy[bi_copy>imax]=imin/200
bi_copy= bi_copy>imin
bi_copy=scp.ndimage.binary_fill_holes(bi_copy)
bi_copy=sk.morphology.binary_opening(bi_copy,sk.morphology.cube(2))
bi_copy=bi_copy*mask[0]
bi_copy=scp.ndimage.binary_fill_holes(bi_copy)
bi_copy=sk.morphology.opening(bi_copy,sk.morphology.cube(4))
bi_copy=sk.morphology.closing(bi_copy,sk.morphology.cube(4))
bi_copy=sk.morphology.remove_small_objects(bi_copy,min_size=20)
binarised=np.copy(bi_copy)
#plt.imshow(bilateral[0]),plt.imshow(binarised[0],alpha=0.4,cmap='gray')
binarised=binarised+1


rw=sk.segmentation.random_walker(bilateral,binarised,beta=100,mode='cg')
plt.figure(),plt.imshow(bilateral[0],cmap='gray'),plt.contour(rw[0],[1]),plt.show()
#skimage.restoration.denoise_bilateral(image, win_size=None,
# sigma_color=None, sigma_spatial=1, bins=10000, mode='constant',
# cval=0, multichannel=False)
def filter_compare(dat,bilateral):
    hi_dat=sk.exposure.histogram(dat)
    hi_bilateral=sk.exposure.histogram(bilateral)
    fig=plt.figure(figsize=(12,12))
    plt.subplot(221),plt.imshow(dat[0])
    plt.subplot(222),plt.imshow(bilateral[0])
    plt.subplot(212),plt.plot(hi_dat[1],hi_dat[0],label="img"),
    plt.plot(hi_bilateral[1],hi_bilateral[0],label="bilateral")
    plt.legend(),plt.figure()