# file to visualise the voxels selected at the interface
# written by Gaêl Ginot @ ICS, Strasbourg, 22 oct 2021

import pyfiltering.bodycharacterisation as bdch
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab 
import csv
from scipy.spatial import Delaunay
#%% goal is to show the contact network only

cof_1 = "first_body.out"
pos1 = list()
with open(cof_1,'r') as csvfile:
    data = csv.reader(csvfile,delimiter=' ')
    for row in data:
        pos1.append([row[1],row[2],row[3]])
csvfile.close()
pos1 = np.asarray([[float(x[0]),float(x[1]),float(x[2])] for x in pos1])

cof_2 = "second_body.out"
pos2 = list()
with open(cof_2,'r') as csvfile:
    data = csv.reader(csvfile,delimiter=' ')
    for row in data:
        pos2.append([row[1],row[2],row[3]])
csvfile.close()
pos2 = np.asarray([[float(x[0]),float(x[1]),float(x[2])] for x in pos2])

#%%this part shows the full contours of the drops 11 and 12
def show_dc(targets,volumes,Centroids,fname,step=100):#only droplet contours and centroids
    targets=[x for x in targets if x in volumes[:,0]]
    mlab.figure(bgcolor=(0,0,0))
    #step=100
    C1=bdch.ContourFinder(str(targets[0]),"../SlicesY_Contours/contours.csv")
    C2=bdch.ContourFinder(str(targets[1]),"../SlicesY_Contours/contours.csv")
    C1 = np.asarray([y.astype(np.uint16) for y in C1])
    C2 = np.asarray([y.astype(np.uint16) for y in C2])
    centroid_1,centroid_2=bdch.centroid(C1),bdch.centroid(C2)
    indices_1=bdch.points_in_cylinder(centroid_1,centroid_2,20,C1)
    indices_2=bdch.points_in_cylinder(centroid_2,centroid_1,20,C2)
    cap_1,cap_2 = C1[indices_1],C2[indices_2]
    droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
    mlab.plot3d(C1[:,0][0::step],C1[:,1][0::step],C1[:,2][0::step],color=droplet_color)
    droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
    mlab.plot3d(C2[:,0][0::step],C2[:,1][0::step],C2[:,2][0::step],color=droplet_color)
    mlab.points3d(cap_1[:,0],cap_1[:,1],cap_1[:,2],color=(1,0,0))
    mlab.points3d(cap_2[:,0],cap_2[:,1],cap_2[:,2],color=(1,0,0))
##    for x in targets:    
#        C=bdch.ContourFinder(str(x),fname)
#        C=np.asarray([y.astype(np.uint16) for y in C])
#        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
#        mlab.plot3d(C[:,0][0::step],C[:,1][0::step],C[:,2][0::step],color=droplet_color)
#    for x in targets:
#        C=Centroids[Centroids[:,0]==x]
#        mlab.points3d(C[0],C[1],C[2],scale_factor=20)
#    C1=ContourFinder(str(ii),input_folder+"contours.csv")
#    C2=ContourFinder(str(jj),input_folder+"contours.csv")
    

        
volumes=bdch.file_import("../SlicesY_Contours/contours_volume.csv")
Centroids=bdch.file_import("../SlicesY_Contours/centroids.csv")
show_dc([210,211],volumes,Centroids,"../SlicesY_Contours/contours.csv",step=100)

#%%
#mlab.figure(bgcolor=(0,0,0))
#mlab.plot3d(pos1[0:-1:20,0],pos1[0:-1:20,1],pos1[0:-1:20,2],representation='surface',color=(1,0,0))
#mlab.plot3d(pos2[0:-1:20,0],pos2[0:-1:20,1],pos2[0:-1:20,2],representation='points',color=(1,0,0))
