#from pyfiltering.imagefiltering import *
import pyfiltering.imagefiltering as imf
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os 
import scipy as scp
import skimage as sk
import skimage.morphology, skimage.segmentation, skimage.measure #don't know why, but need to be imported this way
#Sometimes, outer circle is not removed. How can I remove it for sure ?
i_folder="./DatasICS/SliceYDroplets10um/SlicesY_beamcorrected/"
N_img=100
dat=imf.StackImportation(i_folder,N_img,step=1,skip=200)
dat=imf.PictureNormalisation(dat)
##!! if needed, to show the intensity histogram
#h1=sk.exposure.histogram(dat)
#plt.plot(h1[1],h1[0])
##!!
intensity_kernel,spatial_kernel=0.15,4
bilateral=imf.BilateralFiltering(dat,intensity_kernel,spatial_kernel)
#del dat
bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
##!! if needed, to show the intensity histogram
#h1=sk.exposure.histogram(bilateral[2]),plt.plot(h1[1],h1[0])
##!!
cropped=imf.StackCropping(np.copy(bilateral),0.2)[0]
#!!New test for segmentation
bi_copy=np.copy(dat)
imin,imax=0.51,0.75
bi_copy[bi_copy>imax]=imin/2
bi_copy= bi_copy>imin
#bi_copy=sk.morphology.closing(bi_copy,sk.morphology.ball(1))
#bi_copy=sk.morphology.closing(bi_copy,sk.morphology.cube(1))
bi_copy=bi_copy+1
rw=sk.segmentation.random_walker(bilateral,bi_copy,beta=1000,mode='cg')
#!!
#rw=imf.StackSegmentation(bilateral,threshold=0.2,imin=0.555,imax=0.765)
#del bilateral
rw=sk.morphology.closing(rw==2,sk.morphology.cube(3))
#rw=1-sk.morphology.remove_small_holes(rw==1,area_threshold=20)
rw=sk.morphology.opening(rw==1,sk.morphology.ball(3))
#rw=1-sk.morphology.remove_small_objects(rw==1,min_size=2000)
plt.imshow(rw[0]),plt.show()
plt.imshow(rw[1]),plt.show()
k=1
plt.figure(),plt.imshow(cropped[k],cmap='gray'),plt.contour(rw[k],[1]),plt.show()
labels= sk.measure.label(rw,background=0,connectivity=3)
#vx_size=[np.sum(labels==np.unique(labels)[i]) for i in range(len(np.unique(labels)))]
min_size=100*N_img
labels=sk.morphology.remove_small_objects(labels,min_size=min_size)
plt.imshow(labels[0])
labels = (labels + 1).astype(np.uint8)
#del rw,cropped
#%% Storing the label array
o_file="./DatasICS//SliceYDroplets10um/label_file.txt"
with open(o_file, 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(labels.shape))
    for data_slice in labels:
        np.savetxt(outfile, data_slice, fmt='%-7.2f')
        outfile.write('# New slice\n')
#%% another way to store the label array would be in the format [label value,x,y,z] for every point that is not background
q=[[None,None,None,None]]*np.sum(labels[0]!=1)
for i in range(len(labels[0])):
    for j in range(len(labels[0][i])):
        if labels[0][i][j]!=1:
            q[k]=[labels[0][i][j],0,i,j]
            k=k+1
#q=np.asarray
#o_file_s="./DatasICS//SliceYDroplets10um/label_file_shortened.txt"
#with open(o_file_s,'w') as outfile:
#    outfile.write('# Array shape: {0}\n'.format(labels.shape))
#    for data_slice in labels:
        
#%% is everything okay with the labels ?
def LabelCheck(label):
    if len(np.unique(label))!=np.max(label):
        print("Some labels are missing")
def LabelNumber(label):
    return len(np.unique(label))
def LabelCompare(label1,label2):
    out=[x for x in np.unique(label1) if x not in np.unique(label2)]
    out2=[x for x in np.unique(label2) if x not in np.unique(label1)]
    return out,out2
#%% it looks like mayavi only shows a body if it's present on every slice ?
from mayavi import mlab
mlab.figure(bgcolor=(0, 0, 0))
s = mlab.pipeline.scalar_field(labels)
contour = mlab.pipeline.contour(s)
smooth = mlab.pipeline.user_defined(contour, filter='SmoothPolyDataFilter')
smooth.filter.number_of_iterations = 400
smooth.filter.relaxation_factor = 0.015
curv = mlab.pipeline.user_defined(smooth, filter='Curvatures')

surf = mlab.pipeline.surface(curv)
module_manager = curv.children[0]
module_manager.scalar_lut_manager.data_range = np.array([-0.6,  0.5])
module_manager.scalar_lut_manager.lut_mode = 'RdBu'
mlab.show()
#import skimage.io
#sk.io.imshow('./mayavi_screenshot.png')
#%%
from mayavi import mlab
l1,l2,l3=len(labels),len(labels[0]),len(labels[0][0])

X,Y,Z = np.mgrid[0:l1, 0:l2,0:l3]
mlab.contour3d(X,Y,Z,labels)
mlab.show()
#%% compute distance between particles
#label2=(labels==2)
x,y,z=np.where(labels==2)
c2=np.array([x,y,z])
centroid=[np.mean(x),np.mean(y),np.mean(z)]
#label3=(labels==3)
x3,y3,z3=np.where(labels==3)
c3=np.array([x3,y3,z3])
centroid3=[np.mean(x3),np.mean(y3),np.mean(z3)]
distance = (c3[0]-centroid[0])**2+(c3[1]-centroid[1])**2+(c3[2]-centroid[2])**2
thresh= distance<(np.mean(distance)-1.5*np.std(distance))
print(len(distance)/np.sum(thresh))
c3=[x for x in c3 if thresh[np.where(x==c3)]==True]
#%%
            #code image rendering into vapory
            #import vapory as vap
            #camera = vap.Camera( 'location', [0, 2, -3], 'look_at', [0, 1, 2] )
            #light = vap.LightSource( [2, 4, -3], 'color', [1, 1, 1] )
            #sphere = vap.Sphere( [0, 1, 2], 2, vap.Texture( vap.Pigment( 'color', [1, 0, 1] )))
            #scene = vap.Scene( camera, objects= [light, sphere] )
            #scene.render(labels[0], width=400, height=300 )

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