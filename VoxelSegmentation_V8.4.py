import pyfiltering.imagefiltering as imf
import pyfiltering.bodycharacterisation as bdch
import numpy as np
import matplotlib.pyplot as plt

import skimage as sk
import skimage.morphology, skimage.segmentation, skimage.measure 


N_img,step,header,imin,imax=100,1,10,0.48,0.72
intensity_kernel,spatial_kernel=0.1,4
i_folder="./DatasICS/SliceYDroplets10um/SlicesY_beamcorrected/"
#i_folder="./DatasAGius/Ech1/Cropped_ech1/"
dat=imf.StackImportation(i_folder,N_img,step=step,skip=header)
dat=imf.PictureNormalisation(dat)
bilateral=imf.BilateralFiltering(dat,intensity_kernel,spatial_kernel)
bilateral=imf.PictureNormalisation(bilateral,0.,100.0)
cropped=imf.StackCropping(np.copy(bilateral),0.2)[0]
bi_copy=np.copy(dat)
del dat
bi_copy[bi_copy>imax]=imin/2
bi_copy= bi_copy>imin
bi_copy=bi_copy+1
rw=sk.segmentation.random_walker(bilateral,bi_copy,beta=1000,mode='cg')
rw=sk.morphology.closing(rw==2,sk.morphology.cube(3))
rw=sk.morphology.opening(rw==1,sk.morphology.ball(3))

k=1
plt.figure(),plt.imshow(cropped[k],cmap='gray'),plt.contour(rw[k],[1]),plt.show()
labels= sk.measure.label(rw,background=0,connectivity=3)
min_size=100*N_img
labels=sk.morphology.remove_small_objects(labels,min_size=min_size)
plt.imshow(labels[0])
labels = (labels + 1).astype(np.uint8)

#Uncomment this part if you want to overwrite the output files
#imf.Labels(labels,fname="./DatasICS/SliceYDroplets10um/Volumes.txt")
#imf.ContourFinding(labels,fname="./DatasICS/SliceYDroplets10um/Contours.txt")

vol = bdch.BodyVolume("./DatasICS/SliceYDroplets10um/Volumes.txt",labels=labels) #if labels is loaded in memory, uses it instead of reading filename
rad = [(3.*x/4./np.pi)**(1./3.) for x in vol]