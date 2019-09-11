import imagefiltering as imf
#from PIL import Image
#import numpy as np
import matplotlib.pyplot as plt
#%%
i_folder="../DatasICS/SliceYDroplets10um/SlicesY/"
o_folder="../DatasICS/SliceYDroplets10um_filtered/"
dat=imf.StackImportation(i_folder,5)
dat=imf.PictureNormalisation(dat)
dat_copy=dat

dat=imf.StackRadialCorrection(dat)
dat=imf.BilateralFiltering(dat)
dat=imf.PictureNormalisation(dat)
plt.imshow(dat[1], cmap='gray'),plt.colorbar(),plt.show()
imf.StackSave(dat,i_folder,o_folder)

#%%

from scipy import ndimage
from skimage import morphology

sample = dat_copy[0] > 0.75
sample = ndimage.binary_fill_holes(sample)
open_object = morphology.opening(sample, morphology.ball(3))
close_object = morphology.closing(open_object, morphology.ball(3))
plt.imshow(sample, cmap='gray')
bbox = ndimage.find_objects(close_object)
mask = close_object[bbox[0]]


roi_dat = dat[bbox[0]]
plt.imshow(roi_dat[0], cmap='gray')
plt.contour(mask[0])