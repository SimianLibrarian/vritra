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
folder="../DatasAGius/Ech1/Reconstructed_ech1/"
x,y,w,h=vr.crop(folder)


for i in range(15):
    fname=os.listdir(folder)[i*10]
    img=cv2.imread(folder+fname)
    rs=img[y+5:y+h-5,x+5:x+w-5]
    if i==0:
        vr.detection_silent(rs,'CirclesPositions.txt',mode='w',filename=fname)
    else:
        vr.detection_silent(rs,'CirclesPositions.txt',filename=fname)
#%%
pos=np.genfromtxt('CirclesPositions.txt',usecols=(1,2))
label=np.genfromtxt('CirclesPositions.txt',usecols=0)
r7=pos[label==np.unique(label)[7]]

r8=pos[label==np.unique(label)[8]]

fig=plt.figure(figsize=(8,8))
plt.plot(pos[:,0],pos[:,1],'o',markersize=1.5,color='red')
plt.show()

#r8[abs(r8[:,0]-r7[0,0])/r7[0,0]<0.05 and abs(r8[:,1]-r7[0,1])/r7[0,1]<0.05]

#This part takes all the informations, and will first look for 
#a picture with more than 30 particles detected
#Then it will look for particles that are less than 1% of pic width away
#center-wise, considering the datas from the file (that are minimal enclosing circles
# centers and positions)
#after all, it will output the pic number, the index of this center in
#the classification of its own picture, and the center
#now a future improvement will be to add the contour in the file
#but how can I do it ?

k=0
p=0.1
for i in range(len(np.unique(label))):
    if len(pos[label==np.unique(label)[i]])>30 and k==0:
        k,r=i,pos[label==np.unique(label)[i]]
        g=len(r)-1
        print(int(np.unique(label)[i],g,r[g]))
    elif len(pos[label==np.unique(label)[i]])>30:
        rbis=pos[label==np.unique(label)[i]]
        q=rbis[(abs(rbis[:,0]-r[g,0])/r[g,0]<p) & (abs(rbis[:,1]-r[g,1])/r[g,1]<p)]
        while len(q)>1:
            p=p/2
            q=q[(abs(q[:,0]-r[g,0])/r[g,0]<p) & (abs(q[:,1]-r[g,1])/r[g,1]<p)]
        p=0.1
        if len(q)>0:
            print(int(np.unique(label)[i]),np.where(rbis==q)[0][0],q)






#%%
## import numpy as np
#cnts=cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#for i in range(len(cnts[-1][0])):
#    if np.all(cnts[-1][0,i,0:2]==[-1,-1]):
#        print(i)
#        break
    








