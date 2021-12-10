import pyfiltering.imagefiltering as imf
import pyfiltering.bodycharacterisation as bdch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as scp
import scipy.stats
import skimage as sk
from skimage.feature import peak_local_max
import progressbar
import csv
from mayavi import mlab 
from scipy.spatial import Delaunay


#Function to show the contours of the drops in 3D
def show_d(targets,Contours,step=100):
    mlab.figure(bgcolor=(0,0,0))
    for ii in targets:
        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
        C = Contours[Contours[:,0]==ii]
        mlab.plot3d(C[:,1][0::step],C[:,2][0::step],
                    C[:,3][0::step],color=droplet_color,
                    representation='points')
  
#Function to show the contours of the drops in 3D and their contacts
def show_nc(targets,Contours,Centroids,ContactPairs,ccolor=(1,0,0),step=20):
    mlab.figure(bgcolor=(0,0,0))
    for ii in targets:
        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
        C = Contours[Contours[:,0]==ii]
        mlab.plot3d(C[:,1][0::step],C[:,2][0::step],
                    C[:,3][0::step],color=droplet_color,
                    representation='points')
    target_contacts=[x for x in ContactPairs if x[0] in targets and x[1] in targets]
    for i in range(len(target_contacts)):
        cp_target=target_contacts[i]
        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
        mlab.plot3d([v1[1],v2[1]],[v1[2],v2[2]],[v1[3],v2[3]],
                            color=(1,0,0),tube_radius=2)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def max_dist(a):#return the maximal distance between two points in an array
    d=0
    for i in range(len(a)):
        for j in range(i):
            dist = ((a[i][0]-a[j][0])**2+(a[i][1]-a[j][1])**2+(a[i][2]-a[j][2])**2)**0.5
            if dist>d:
                d=dist
    return d


def pair_list_init(n):
    nf=np.math.factorial(n)/np.math.factorial(n-2)/2
    pair_indices,k = [[0,0]]*nf,0
    for i in range(n):
        for j in range(i):
            pair_indices[k]=[i,j]
            k=k+1
    return pair_indices

def pair_distances(centroid):
    d = list()
    for i in range(len(centroid)):
        for j in range(i):
            d.append(((centroid[i][0]-centroid[j][0])**2+(centroid[i][1]-centroid[j][1])**2+(centroid[i][2]-centroid[j][2])**2)**0.5)
    return d

#merging seeds when theyu are too close to each others
def centroid_merging(centroid,d0):   
    merged="no"
    if np.size(centroid)/3<2:
        return centroid
    else:
        d = pair_distances(centroid)
        n=np.size(centroid)/3
        nf=np.math.factorial(n)/np.math.factorial(n-2)/2
        pair_indices,k = [[0,0]]*nf,0
        for i in range(n):
            for j in range(i):
                pair_indices[k]=[i,j]
                k=k+1         
        for i in np.argsort(d):
            if d[i]<d0:
                ii = pair_indices[i][0]
                jj = pair_indices[i][1]
                centroid[ii]=[(centroid[ii][0]+centroid[jj][0])/2.,
                        (centroid[ii][1]+centroid[jj][1])/2.,
                        (centroid[ii][2]+centroid[jj][2])/2.]
                centroid =np.delete(centroid,jj,axis=0)
                merged="yes"
                break
        return centroid,merged
#%%
    
i_folder="./SlicesY/" #folder where the initial slices are found
o_folder="./SlicesY_BilateralFiltered/" #folder where filtered slics will be saved
#step : number of images skipped between every imported slice
step = 1
#header : number of slices skipped at the bottom of the stack
header = 0
#N_img : number of images imported (here declared as the total number of slices)
N_img=len(os.listdir(i_folder))

#standard deviations of the two gaussian filters of the bilateral filter
intensity_kernel,spatial_kernel=0.01,0.2


#In this section, the program checks if the slices have been already filtered with these parameters
#if the parameters of the last filter (stored in o_folder/FilterParameters.txt) are different, it proposes to filter again with the new parameters 
#caution : refiltering will erase the old filtered slices !

#part to check if the bilateral filter has already been applied
bfo = [x for x in os.listdir(o_folder) if '.tif' not in x]
if len(bfo)>0:
    bfo = np.genfromtxt(o_folder+bfo[0])
    if bfo[1][3]==intensity_kernel and bfo[1][4]==spatial_kernel:
        print("Bilateral filter has already been applied. Skipping the filter step.\n")
    else:
        print("Bilateral filter already applied with different parameters.\n")
        answer = input("Do you want to filter with new parameters ? (previous datas will be erased) [Y/n]\n")
        if answer=='Y':
            print("Applying bilateral filter. Erasing previous images.\n")
            #Importation of the images from the unfiltered directory
            dat=imf.StackImportation(i_folder,N_img,step=step,skip=header)
            #Application of the bilateral filter
            bilateral=imf.BilateralFiltering2D(dat,intensity_kernel=intensity_kernel,
                                               spatial_kernel=spatial_kernel)
            #Stack rotation (xyz)->(yzx)
            temp=np.transpose(np.asarray(bilateral),(1,2,0))
            #Application of the bilateral filter
            temp=imf.BilateralFiltering2D(temp,intensity_kernel=intensity_kernel,
                                               spatial_kernel=spatial_kernel)
            #Stack rotation (yzx)->(zxy)
            temp=np.transpose(np.asarray(temp),(1,2,0))
            #Application of the bilateral filter
            temp=imf.BilateralFiltering2D(temp,intensity_kernel=intensity_kernel,
                                               spatial_kernel=spatial_kernel)
            #Stack rotation (zxy)->(xyz)
            temp=np.transpose(np.asarray(temp),(1,2,0))
            #reduction in a 16 bits data type
            bilateral= temp.astype(np.uint16)
            #Saving of fitlered slices in o_folder
            imf.ImageSaving(bilateral,o_folder,N_img,step,header,
                    intensity_kernel,spatial_kernel)  

else:#if no parameter file is present, that means that no filtering has been done
    print("Applying bilateral filter for the first time.\n")
    dat=imf.StackImportation(i_folder,N_img,step=step,skip=header)
    bilateral=imf.BilateralFiltering2D(dat,intensity_kernel=intensity_kernel,
                                   spatial_kernel=spatial_kernel)
    temp=np.transpose(np.asarray(bilateral),(1,2,0))
    temp=imf.BilateralFiltering2D(temp,intensity_kernel=intensity_kernel,
                                   spatial_kernel=spatial_kernel)
    temp=np.transpose(np.asarray(temp),(1,2,0))
    temp=imf.BilateralFiltering2D(temp,intensity_kernel=intensity_kernel,
                                   spatial_kernel=spatial_kernel)
    temp=np.transpose(np.asarray(temp),(1,2,0))
    bilateral= temp.astype(np.uint16)
    imf.ImageSaving(bilateral,o_folder,N_img,step,header,
        intensity_kernel,spatial_kernel)

#############################
#### Picture binarisation ###
#############################

# imin and imax are the (normalised) intensities taken as lower (resp. upper) limit of the intensities corresponding to drops/bubbles
# finding the good binarisation parameters requires trial and error
imin,imax=0.5,0.95

#oo_folder is the folder where the slices with labelled drops are stored
oo_folder="./SlicesY_Labelled/"

#In this section, the program checks if the slices have been already binarised with these parameters
#if the parameters of the last filter (stored in oo_folder/BinarisationParameters.txt) are different, it proposes to filter again with the new parameters 
#caution : rebinarising will erase the old binarised slices !
lfo = [q for q in os.listdir(oo_folder) if '.tif' not in q]
if "BinarisationParameters.txt" in lfo:
    lfo = np.genfromtxt(oo_folder+"BinarisationParameters.txt")
    if imin==np.min(lfo) and imax==np.max(lfo) and len([q for q in os.listdir(oo_folder) if '.tif' in q])==N_img:
        print("\nAlready binarised. Skipping this step.\n")
    elif imin==np.min(lfo) and imax==np.max(lfo) and len([q for q in os.listdir(oo_folder) if '.tif' in q])<N_img:
        print("\n Not enough pictures binarised. Binarising again.\n")
        try:
            bilateral
            if len(bilateral)!=N_img:
                print("\n Wrong number of slices. Loading images again.\n")
                bilateral=imf.StackImportation(o_folder,N_img,step=step,skip=header)
        except NameError:
            print("\nNo filtered images loaded. Loading again.\n")
            bilateral=imf.StackImportation(o_folder,N_img,step=step,skip=header) #note : il semble que ce soit le transfert hors de la fonction qui ralentisse le système
        #this part sets to black all voxels outside of a circle centered in the middle of the horizontal slice
        #it is here because of the shape of the samples I studied : you may want to change it depending on the shape of your sample container
        mask = create_circular_mask(len(bilateral[0]),
                    len(bilateral[0][0]),radius=len(bilateral[0])*0.95*0.5)
        
        #binarisation : everything above imax and below imin is set to 0, else to 1
        binarised = np.asarray([(q*mask).astype(np.uint16) for q in bilateral])
        binarised=imf.PictureNormalisation(binarised,m=10,M=95)
        binarised[binarised>imax]=imin/2
        binarised= binarised>imin
        binarised = binarised+1
        #going from (0,1) to (1,0) to adapt to the segmentation algorithm
        binarised=binarised+1
        #segmentation 
        binarised=sk.segmentation.random_walker(binarised,binarised,beta=1000,mode='bf')
        rw = sk.measure.label(binarised,background=1,connectivity=3)
        
        rw = np.asarray([(q==2)+1 for q in rw])
        print("Saving binarised images in Labelled directory.\n")
        imf.ImageSaving(rw+1,oo_folder,N_img,step,header,intensity_kernel,spatial_kernel)
        
        #Saving the binarisation parameters
        with open(oo_folder+"BinarisationParameters.txt","w") as csvfile:
            csvfile.write("#Imin Imax\n")
            csvfile.write(str(imin)+" "+str(imax)+"\n")
            csvfile.close()
else:
    print("\nFirst time binarisation.\n")
    try:
        bilateral
        if len(bilateral)!=N_img:
            print("\n Wrong number of slices. Loading images again.\n")
            bilateral=imf.StackImportation(o_folder,N_img,step=step,skip=header)
    except NameError:
        print("\nNo filtered images loaded. Loading again.\n")
        bilateral=imf.StackImportation(o_folder,N_img,step=step,skip=header) #note : il semble que ce soit le transfert hors de la fonction qui ralentisse le système
    
    mask = create_circular_mask(len(bilateral[0]),
                len(bilateral[0][0]),radius=len(bilateral[0])*0.95*0.5)
    
    binarised = np.asarray([(q*mask).astype(np.uint16) for q in bilateral])
    binarised=imf.PictureNormalisation(binarised,m=10,M=95)
    binarised[binarised>imax]=imin/2
    binarised= binarised>imin
    binarised = binarised+1
    binarised=binarised+1
    binarised=sk.segmentation.random_walker(binarised,binarised,beta=1000,mode='bf')
    rw = sk.measure.label(binarised,background=1,connectivity=3)
    
    rw = np.asarray([(q==2)+1 for q in rw])
    print("Saving binarised images in Labelled directory.\n")
    imf.ImageSaving(rw+1,oo_folder,N_img,step,header,intensity_kernel,spatial_kernel)
    
    with open(oo_folder+"BinarisationParameters.txt","w") as csvfile:
        csvfile.write("#Imin Imax\n")
        csvfile.write(str(imin)+" "+str(imax)+"\n")
        csvfile.close()

##############################################
### Detection of seeds for watersheding ######
##############################################
        
print("\n Importation of labeled images again for watersheding.\n")
labels = imf.StackImportation(oo_folder,N_img,step=1,skip=header)
labels = labels.astype(np.uint16)
print("\n Finding the positions of the seeds for watersheding.\n")
try:
    Seeds
except NameError:
    Seeds = np.zeros(np.shape(labels))
    if 'peaks.out' in os.listdir("./"):
        print("\nPeak file already existing. Importing.\n")
        with open("peaks.out","r") as csvfile:
            spamreader = csv.reader(csvfile,delimiter='\t')
            Peak = list()
            for row in spamreader:
                Peak.append(row)
            Peak = np.asarray(Peak).astype(np.uint16)
        Peak = np.asarray(Peak).astype(np.uint16)
        m=0       
        for q in Peak:
            Seeds[q[2]-1,q[0]-1,q[1]-1]=m
            m=m+1
        Seeds = np.asarray(Seeds).astype(np.uint16)
    else:
        print("\nPeaks not defined. Computing.\n")
        Peak = list()
        for k in np.unique(rw):
            if k!=0:
#Computation of the euclidean distance map for seed detection
                edm_one = sk.morphology.erosion(rw==k,sk.morphology.cube(10)) #removing outer voxels for easier segmentation
                edm_one = np.asarray([scp.ndimage.morphology.distance_transform_edt(q) for q in edm_one]) #applying euclidean distance map computation
                edm_one = scp.ndimage.filters.maximum_filter(edm_one*(edm_one>20),size=5) #removing the 20 outer layers
                print("Segmentation.\n")
                #labelling the different bodies where we try to find the seeds
                rw2 = sk.segmentation.random_walker((edm_one>0)+1,(edm_one>0)+1) 
                rw2 = sk.measure.label(rw2,background=1,connectivity=3)
                rw2 = np.asarray(rw2).astype(np.uint16)
                print("Peak detection.\n")
                #detection of the local maxima with only one per label
                peak = peak_local_max(edm_one,labels=rw2,num_peaks_per_label=1)
                #deleting variables to free RAM memory
                del(rw2)
                del(edm_one)
                print("Peak adding.\n")
                for q in peak:
                    Peak.append(q)
        Peak = np.asarray(Peak).astype(np.uint16) #final position of the seeds
        #Saving the seeds positions in peaks.out file
        with open("./peaks.out","w") as csvfile:
            peakwriter = csv.writer(csvfile,delimiter="\t")
            for i in range(len(Peak)):
                peakwriter.writerow([int(Peak[i][1]),int(Peak[i][2]),int(Peak[i][0])])
        csvfile.close()
        #creating a Seeds 3D array with the labels of the seeds at their position
        m=0
        for q in Peak:
            Seeds[q[2]-1,q[0]-1,q[1]-1]=m
            m=m+1
        Seeds = np.asarray(Seeds).astype(np.uint16)
            
#####################################################
################ Watersheding #######################
#####################################################
        
try:
    labels
except NameError:
    labels = imf.StackImportation(oo_folder,N_img,step=1,skip=header)
    labels = labels.astype(np.uint16)

#Computation of the euclidean distance map using for the flooding (different of the one used for seed detection)        
edm_ws = scp.ndimage.morphology.distance_transform_edt(labels-2)

edm_ws = np.max(edm_ws)-edm_ws
edm_ws = np.asarray(edm_ws).astype(np.uint16)

#flooding the basins, using the seeds positions as seeds, the edm as basins, and the labels as mask to limit the flooding in the drops
basins = sk.segmentation.watershed(edm_ws,markers=Seeds,mask=labels-2)

#Display of the result to check the quality of the segmentation
plt.figure(figsize=(10,10)),plt.imshow(basins[0])

#Saving the slices with segmented drops labelled (label as pixel value)
imf.ImageSaving(basins,"./SlicesY_Labelled_Watershed/",len(basins),1,0,0,0)

#Saving the volume of every drop in a (label,number of voxels) format
bdch.VolumesWriter(basins,"./SlicesY_Contours/"+"contours_volume.csv")
#Saving the position of contour voxels in a (label,x,y,z) format
bdch.ContourWriter(basins,"./SlicesY_Contours/"+"contours.csv")


#Writing the positions of the centroids as the average position of the contour voxels
try:
    contours
except NameError:
    c_folder = "./SlicesY_Contours/"
    co_file = "contours.csv"
    contours=bdch.file_import(c_folder+co_file)

ce_file="centroids.csv"
bdch.centroid_writing(contours,c_folder+ce_file)

try:
    centroids
except NameError:
    centroids = bdch.file_import(c_folder+ce_file)

###################################
# Showing reconstructed emulsion ##
###################################

show_d([q[0] for q in centroids],contours,step=20)

###################################
#### S2S distance computation #####
###################################
    
centroids = np.asarray([q for q in centroids])
Pairs = bdch.delaunay_edges(centroids)
os.chdir("./s2s/")
with progressbar.ProgressBar(max_value=len(Pairs)) as bar:
    k=0
    for ii in Pairs:
        bar.update(k)
        k=k+1
        if "s2s_"+str(min(ii[0],ii[1]))+"_"+str(max(ii[0],ii[1]))+".out" not in os.listdir("./distances/"):
            os.system("./cap_writer.exe "+str(ii[0])+" "+str(ii[1])+" && ./s2s.exe > ./distances/s2s_"+str(min(ii[0],ii[1]))+"_"+str(max(ii[0],ii[1]))+".out")
            
os.chdir("../")  

###################################
##### distance summarising ########
###################################
su_file="./SlicesY_Contours/distances.csv"
cs_folder="./s2s/distances/"
content = os.listdir(cs_folder)
content = [x for x in content if ".out" in x]

with open(su_file,'a') as csvfile:
    spamwriter = csv.writer(csvfile,delimiter=';')
    with progressbar.ProgressBar(max_value=len(content)) as bar:
        k=0
        for i in content:
            bar.update(k)
            k=k+1
            n1,n2 = i[i.find('_')+1:i.rfind('_')],i[i.rfind('_')+1:i.find('.')] #n1,n2 are the indices of the pair
            t = np.genfromtxt(cs_folder+i)
            if len(t)>0:
                #saving the 5th centile of the S2S distances as the actual S2S distance
                # format : (label 1, label 2, distance S2S)
                spamwriter.writerow([n1,n2,np.percentile(t,5)])

csvfile.close()

#####################################
# Application of distance threshold #
#####################################

su_file="./SlicesY_Contours/distances.csv"
cs_folder="./s2s/distances/"
Pairs_dist = list()
with open(su_file,'r') as csvfile:
    reader = csv.reader(csvfile,delimiter=';')
    for row in reader:
        Pairs_dist.append([int(row[0]), int(row[1]), float(row[2])])
csvfile.close()
Pairs_dist = np.asarray(Pairs_dist)
d_thresh = 15
Pairs_dist = [[int(q[0]),int(q[1])] for q in Pairs_dist if q[2]<d_thresh]

#####################################
## Display of the contact network ###
#####################################

show_nc([q[0] for q in centroids],contours,centroids,Pairs_dist,"./SlicesY_Contours/contours.csv",step=20)
for angle in range(0,360,5):
    mlab.view(azimuth=angle)
    mlab.savefig("./GifMaker/Projection_"+str(angle)+".png",size=(1000,1000))
