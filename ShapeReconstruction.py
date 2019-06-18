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
#%%
from pyimagesearch.volumereconstructor import VolumeReconstructor
vr=VolumeReconstructor()
#vr.detection_verbose('./Giustiniani/img_1909.tif')
vr.detection_silent('./Giustiniani/img_1909.tif','CirclesPositions.txt',mode='w')


#%% part for getting the circle from the tomographic reconstruction
#The window parameter are computed with the first picture, then taken as constants for the next ones
import cv2
folder="../Données Anaïs Giustiniani/Ech1/Reconstructed_ech1/"
x,y,w,h=vr.crop(folder)


for i in range(15):
    img=cv2.imread(folder+os.listdir(folder)[i*10])
    rs=img[y+5:y+h-5,x+5:x+w-5]
    vr.detection_silent(rs,'CirclesPositions.txt')
#%%
## import numpy as np
#cnts=cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#for i in range(len(cnts[-1][0])):
#    if np.all(cnts[-1][0,i,0:2]==[-1,-1]):
#        print(i)
#        break
    








