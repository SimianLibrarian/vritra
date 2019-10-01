import numpy as np
#import scipy as scp
#import scipy.stats
import random as random
import os
from xml.dom import minidom
#from PIL import Image
#import skimage as sk
#import skimage.restoration,skimage.morphology

def BodyImport(i_file,label_choice):
    label_size=os.popen("cat "+i_file+" | grep -E '^"+str(label_choice)+".00\>' | wc -l").read()
    label_size=int(label_size.split()[0])
    body=[[None,None,None]]*label_size
    with open(i_file,'r') as coords:
        k,r=0,0
        for line in coords:
            if "#" not in line and label_choice==[float(x) for x in line.split()][0] :
                body[r]=[float(x) for x in line.split()][1::]
                r=r+1
    coords.close()
    return np.asarray(body)


def BodySection(body='None',label_choice=0,i_file='None',percentage=0.05):
    if i_file!='None':
        body=BodyImport(i_file,label_choice)
    return random.sample(body,int(len(body)*percentage))

def BodyTemp(body,f,label_choice="None",mode="w"):
    with open(f,mode) as coords:
        if mode=="w":
            coords.write("boundary_condition = periodic_cuboidal, boxsx = 100, boxsy = 200, boxsz=200\n")
#            coords.write("#Label x y z r \n")
#            coords.write("#x y z r \n")
            coords.write("#Label x y z\n")
        if label_choice=="None":
            k=0
            for line in body:    
                k=k+1
#                l=str(k)+' '+str(int(line[0]))+' '+str(int(line[1]))+' '+str(int(line[2]))+' '+str(0.001)+'\n'
                l=str(k)+' '+str(int(line[0]))+' '+str(int(line[1]))+' '+str(int(line[2]))+'\n'                
                coords.write(l)
        else:
            for line in body:
#                l=str(label_choice)+' '+str(int(line[0]))+' '+str(int(line[1]))+' '+str(int(line[2]))+' '+str(0.001)+'\n'
                l=str(label_choice)+' '+str(int(line[0]))+' '+str(int(line[1]))+' '+str(int(line[2]))+'\n'                
                coords.write(l)                
    coords.close()


def PomeloRun(body='None',pomelo_path="./pomelo/bin/pomelo",i_file='None',o_file='LastOutput'):
    os.popen(pomelo_path+" -mode SPHEREPOLY -i "+i_file+" -o "+o_file).read()

def centr_org(c1,c2):
    if c1[2]<=c2[2]:
        return c1,c2
    else:
        return c2,c1

def condition_z(pos1,c1,c2):
    c1,c2=centr_org(c1,c2)
    zmin,zmax=c1[2],c2[2]
    p1r=pos1[ pos1[:,2]>=zmin ] 
    p1r=p1r[p1r[:,2]<=zmax]
    return p1r
def condition_x(pos1,c1,c2,L):
    c1,c2=centr_org(c1,c2)
    zmin,zmax=c1[2],c2[2]
    xmin,xmax=c1[0],c2[0]   
    a=(xmin-xmax)/(zmin-zmax)
    b=xmin-zmin*(xmin-xmax)/(zmin-zmax)
    p1r=pos1[pos1[:,0]<a*pos1[:,2]+b+L/2]
    p1r=p1r[p1r[:,0]>a*p1r[:,2]+b-L/2]
    return p1r

def condition_y(pos1,c1,c2,L):
    c1,c2=centr_org(c1,c2)
    zmin,zmax=c1[2],c2[2]
    ymin,ymax=c1[1],c2[1]   
    a=(ymin-ymax)/(zmin-zmax)
    b=ymin-zmin*(ymin-ymax)/(zmin-zmax)
    p1r=pos1[pos1[:,1]<a*pos1[:,2]+b+L/2]
    p1r=p1r[p1r[:,1]>a*p1r[:,2]+b-L/2]
    return p1r

def condition(pos1,c1,c2,L):
    p1r=condition_z(pos1,c1,c2)
    p1r=condition_x(p1r,c1,c2,L)
    p1r=condition_y(p1r,c1,c2,L) 
    return p1r

def distance(r1,r2):
    return ((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)**(0.5)

#    return (r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2
def BodySeparation(pos1,pos2,L,timecount="No"):
    c1=[np.mean(pos1[:,0]),np.mean(pos1[:,1]),np.mean(pos1[:,2])]
    c2=[np.mean(pos2[:,0]),np.mean(pos2[:,1]),np.mean(pos2[:,2])]
    p1r=condition(pos1,c1,c2,L)
    p2r=condition(pos2,c1,c2,L)
    if len(p1r)==0 or len(p2r)==0:
        return None
#        break
    if timecount=="yes" or timecount=="Yes":
        t1=time.time()
    d=[]
    for i in range(len(p1r)):
        for j in range(len(p2r)):
            d.append(distance(p1r[i],p2r[j]))
    if timecount=="yes" or timecount=="Yes":
        t2=time.time()
        print("Time elapsed :"+str(t2-t1))
    return d

def BodyVolume(filename,labels=0):
    if len(labels)==1:
        labels = np.genfromtxt(filename,usecols=0,dtype=int)
    i_folder=filename[0:filename.rfind('/')+1]
    while 'unireconstruction.xml' not in os.listdir(i_folder) and len(i_folder)>1:
        i_folder= i_folder[0:i_folder.rfind('/',0,i_folder.rfind('/'))+1]           
    param_file = minidom.parse(i_folder+'unireconstruction.xml')
    dimensions=param_file.getElementsByTagName('voxelSize')
    X = float(dimensions[0].attributes['X'].value)*1000. #x dimension of a voxel [um]
    Y = float(dimensions[0].attributes['Y'].value)*1000. #y dimension of a voxel [um]
    Z = float(dimensions[0].attributes['Z'].value)*1000. #z dimension of a voxel [um]
    vox_vol = X*Y*Z    
    vol = [np.sum(labels==x)*vox_vol for x in np.unique(labels) if x !=1] # droplets volumes [um^3]
#    rad = [(3.*x/4./np.pi)**(1./3.) for x in vol]
    return vol
#    labels=np.unique(data[:,0])
#    for i in range(labels):
#        data_spec = data[data[:,0]==labels]
#        volume=len(data_spec)*voxel_volume
# filename= './DatasICS/SliceYDroplets10um/'+'label_file_shortened.txt'