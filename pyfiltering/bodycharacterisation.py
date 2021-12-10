import numpy as np
#import scipy as scp
#import scipy.stats
import psutil
import random as random
import os
from xml.dom import minidom
import time
import multiprocessing as mp
from mayavi import mlab 
from scipy.optimize import leastsq
import csv
from scipy.spatial import Delaunay

from PIL import Image
#from PIL import Image
import skimage as sk
import skimage.restoration,skimage.morphology
from scipy.spatial import cKDTree as KDTree
#import boo#from boo import boo

def StackImportation(folder,N,step=1,skip=0):
    content=OrganiseSlices(folder)
#    dat=[None]*N
    dat=list()
    im_frame=Image.open(folder+content[int(skip)])
    size=im_frame.size[1::-1]
    for i in range(N):
        if i!=0:
            im_frame = Image.open(folder+content[int(skip)+i*int(step)])
        dat.append(np.array(im_frame.getdata()).reshape(size).astype(np.int16))
#        print("Table size: "+format_bytes(getsizeof(dat[i])))   
#    return dat
    return np.asarray(dat)
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

def collect_result(result):
    global results
    results.append(result)
def DistanceSelection(dist_array):
    return np.mean(np.percentile(dist_array,0.05))
def g(pos1,i1,pos2,i2,L):
    d = DistanceSelection(BodySeparation(pos1,pos2,L))
#    print(d)
    return [i1,i2,d]

def ParallelSurfaceDistance(cnts,L,step):
    pool=mp.Pool(mp.cpu_count())
    global results
    results =[]
    for i in np.unique(cnts[:,0]):
        for j in np.unique(cnts[:,0]):
            if j>i:
#                print(i,j)
                cnts1=cnts[cnts[:,0]==i][:,1:4]
                cnts2=cnts[cnts[:,0]==j][:,1:4]
                cnts1[:,0]=cnts1[:,0]*step
                cnts2[:,0]=cnts2[:,0]*step
                pool.apply_async(g, args=(cnts1,i,cnts2,j,L), callback=collect_result)
    pool.close()
    pool.join()
    return results


def BodySeparation_FromBonds(pos1,pos2):
#    c1=[np.mean(pos1[:,0]),np.mean(pos1[:,1]),np.mean(pos1[:,2])]
#    c2=[np.mean(pos2[:,0]),np.mean(pos2[:,1]),np.mean(pos2[:,2])]
    d=[]
    for i in range(len(pos1)):
        for j in range(len(pos2)):
            d.append(distance(pos1[i],pos2[j]))
    return d

def DistanceSelection_FromBonds(dist_array):
    return np.mean(np.percentile(dist_array,0.05))
def h(pos1,i1,pos2,i2):
    d = DistanceSelection_FromBonds(BodySeparation_FromBonds(pos1,pos2))
    return [i1,i2,d]
def centroid(C):
    if type(C)==list:
        C=np.asarray(C)
    return np.asarray([np.mean(C[:,0]),np.mean(C[:,1]),np.mean(C[:,2])])

def points_in_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    condition_1=np.where(np.dot(q-pt1,vec)>=0)
    condition_2=np.where(np.dot(q-pt2,vec)<=0)
    condition_3=np.where([np.linalg.norm(np.cross(x-pt1,vec)) for x in q]<=const)
    return [int(x) for x in range(np.max(condition_1)) if x in condition_1[0] and x in condition_2[0] and x in condition_3[0]]

def ParallelSurfaceDistance_FromBonds(cnts,bonds):
    pool=mp.Pool(mp.cpu_count())
    global results
    results =[]
    for i in range(len(bonds)):
        j=bonds[i][0]
        k=bonds[i][1]
        cnts1=cnts[cnts[:,0]==j][:,1:4]
        cnts2=cnts[cnts[:,0]==k][:,1:4]
        centroid_1,centroid_2=centroid(cnts1),centroid(cnts2)
        indices_1=points_in_cylinder(centroid_1,centroid_2,20,cnts1)
        indices_2=points_in_cylinder(centroid_2,centroid_1,20,cnts2)
        cnts1,cnts2=cnts1[indices_1],cnts2[indices_2]
        pool.apply_async(h, args=(cnts1,j,cnts2,k), callback=collect_result)
    pool.close()
    pool.join()
    return results

def elapsed_since(start):
    #return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    elapsed = time.time() - start
    if elapsed < 1:
        return str(round(elapsed*1000,2)) + "ms"
    if elapsed < 60:
        return str(round(elapsed, 2)) + "s"
    if elapsed < 3600:
        return str(round(elapsed/60, 2)) + "min"
    else:
        return str(round(elapsed / 3600, 2)) + "hrs"


def get_process_memory():
    process = psutil.Process(os.getpid())
    mi = process.memory_info()
    return mi.rss, mi.vms, mi.shared


def format_bytes(bytes):
    if abs(bytes) < 1000:
        return str(bytes)+"B"
    elif abs(bytes) < 1e6:
        return str(round(bytes/1e3,2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


def profile(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        rss_before, vms_before, shared_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: {:>20}  RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format("<" + func.__name__ + ">",
                    format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
        return result
    if inspect.isfunction(func):
        return wrapper
    elif inspect.ismethod(func):
        return wrapper(*args,**kwargs)

def OrganiseSlices(folder):
    content=os.listdir(folder)
#    if np.sum(["slice" in content])>=1:
    if len([x for x in content if "slice" in x])>=1:
       content=[x for x in content if "slice" in x]
    if len([x for x in content if "img" in x])>=1:
       content=[x for x in content if "img" in x]
    content_copy=content
    content=[[s for s in content[i]if s.isdigit()] for i in range(len(content))]
    content=[int("".join(content[i])) for i in range(len(content))]
    argsort=np.argsort(content)
    content_copy[argsort[0]]
    content=[content_copy[argsort[i]] for i in range(len(argsort))]
    return content

def ContourFinder(target_label,cfile):
    C=list()
    with open(cfile) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            if target_label==row[0]:
                C.append([float(row[1]),float(row[2]),float(row[3])])
#                C.append([int(row[1]),int(row[2]),int(row[3])])
    return np.asarray(C)
def ContourFinding_slice(l,t,fname="./DatasICS/SliceYDroplets10um/Contours.txt",mode="a"):
    with open(fname,mode) as o_file:
        o_file.write("#New slice "+str(t)+"\n")
        for i in np.unique(l):
            if i!=1:
                cnts = sk.measure.find_contours(l==i,level=0.99)   
                if len(cnts)==1:
                    for j in range(len(cnts[0])):
                        o_file.write(str(i)+" "+str(t)+" "+
                        str(int(cnts[0][j][0]))+" "
                        +str(int(cnts[0][j][1]))+"\n")              
        o_file.close()
def ContourFinding(labels,fname="./DatasICS/SliceYDroplets10um/Contours.txt"):
    for i in range(len(labels)):
        if i==0:
            ContourFinding_slice(labels[i],i,fname=fname,mode='w')
        else:
            ContourFinding_slice(labels[i],i,fname=fname,mode='a')        

def QuickPeek(C1,indices_1,C2,indices_2):
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=black)
    mlab.points3d(C1[:,0],C1[:,1],C1[:,2],color=white,scale_factor=1.2)
    mlab.points3d(C2[:,0],C2[:,1],C2[:,2],color=white,scale_factor=1.2)
    mlab.points3d(C1[indices_1][:,0],C1[indices_1][:,1],
                C1[indices_1][:,2],color=(1,0,0))
    mlab.points3d(C2[indices_2][:,0],C2[indices_2][:,1],
                C2[indices_2][:,2],color=(1,0,0))
    mlab.show()
    
def QuickPeek_Selection(Selection,Centroids,Network,co_name):
    Centroids=np.asarray(Centroids)
#    Centroids=[Centroids[i,1::] for i in range(len(Centroids)) if Centroids[i,0] in Selection]
#    Centroids=np.asarray(Centroids)
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=black)
    for i in range(len(Selection)):
        C1=ContourFinder(str(Selection[i]),co_name)
        centroid_1=centroid(C1)
        indices_1=[]
        #tips : I have a file with only the centroids registered, it will be a lot quicker
        for j in range(len(Selection)):
#            print([Centroids[j][1],Centroids[j][2],Centroids[j][0]])          
            if j!=i:
                tak=np.argwhere(Network==[min(i,j),max(i,j)])
                if len(tak)==len(np.unique(tak[:,0])):
#                    centroid_2=[Centroids[j][1],Centroids[j][2],Centroids[j][0]]
                    centroid_2=Centroids[Centroids[:,0]==Selection[j]]
                    centroid_2=[centroid_2[0][1],centroid_2[0][2],centroid_2[0][3]]
                    indices_1=np.concatenate((indices_1,points_in_cylinder(centroid_1,centroid_2,20,C1)))
                    indices_1=indices_1.astype(int)
                    mlab.points3d(C1[indices_1][:,0][0::30],C1[indices_1][:,1][0::30],
                C1[indices_1][:,2][0::30],color=(1,0,0),scale_factor=6.0)
        mlab.points3d(C1[:,0][0::30],
                      C1[:,1][0::30],C1[:,2][0::30],color=white,scale_factor=2.0)
    mlab.show()

def Angles_Single_Surface(C1,C2,Max='max'):
    centroid_1,centroid_2=centroid(C1),centroid(C2)
    indices_1=points_in_cylinder(centroid_1,centroid_2,20,C1)
    NormalVectors,start=list(),time.time()
    if Max=='max':
        Max=len(indices_1)
    for i in range(0,Max):
        if i%100==0:
            print(str(i)+" over "+str(Max))
            print("Elapsed: "+elapsed_since(start))
        R=C1[indices_1[i]]
        n=normal_vector_v2(C1,indices_1[i])
        if np.dot(n,R-centroid(C1))<0:
            n=-n
        NormalVectors.append(n)
    NormalVectors=np.asarray(NormalVectors)
    NormalVectors=[x for x in NormalVectors if sum(x)!=0]    
    r_reference=centroid_2-centroid_1
    Result=[np.dot(x,r_reference)/np.linalg.norm(r_reference) for x in NormalVectors]
    Result=[np.arccos(abs(x))*180./np.pi for x in Result]
    Result=np.asarray(Result)[np.where(np.isnan(Result)==False)[0]]
    x,y=np.histogram(Result,bins=18,range=[0,90])
    return x,y  

def Angles_In_A_Pair(TargetContacts,cfile,ofile,Max='max',write='yes'):   
    
    C1=ContourFinder(str(TargetContacts[0]),cfile)
    C2=ContourFinder(str(TargetContacts[1]),cfile)
    x1,y1=Angles_Single_Surface(C1,C2,Max=Max)
    if write=="yes":
        with open(ofile,'a') as csvfile:
            datawriter=csv.writer(csvfile,delimiter=";")
            datawriter.writerow(np.concatenate((TargetContacts,x1)))
        csvfile.close()
    x2,y2=Angles_Single_Surface(C2,C1,Max=Max)
    if write=="yes":
        with open(ofile,'a') as csvfile:
            datawriter=csv.writer(csvfile,delimiter=";")
            datawriter.writerow(np.concatenate(([TargetContacts[1],TargetContacts[0]],x2)))
        csvfile.close()
    return x1,y1,x2,y2


def DisplayAngularDistribution(NormalVectors,r_reference):
    Result=[np.dot(x,r_reference)/np.linalg.norm(r_reference) for x in NormalVectors]
    Result=[np.arccos(abs(x))*180./np.pi for x in Result]
    Result=np.asarray(Result)[np.where(np.isnan(Result)==False)[0]]
    plt.hist(Result,bins=18,normed=1,range=[0,90])
    plt.show()

#fit the distribution with a gaussian
def AngularDistribution(Result,r_reference):
    Result=[np.dot(x,r_reference)/np.linalg.norm(r_reference) for x in Result]
    Result=[np.arccos(abs(x))*180./np.pi for x in Result]
    Result=np.asarray(Result)[np.where(np.isnan(Result)==False)[0]]
    y,x=np.histogram(Result,bins=18,range=[0,90])[0:2]
    xdata=[(x[i]+x[i+1])/2. for i in range(len(x)-1)]
    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))
    init  = [np.max(xdata), np.mean(xdata), np.std(xdata), 0]
    out   = leastsq( errfunc, init, args=(xdata, y))
    return out[0][1],out[0][2]



#    return [xfor x in range(np.max(condition_1)) if x in condition_1[0] and x in condition_2[0] and x in condition_3[0]]
  


def IndexShift(r_target,r_test):
    r=[abs(x)<=1 for x in r_target-r_test]
    if sum(r)==3:
        return True
    else:
        return False
def Neighbours(PointsArray,Index):
    d=[np.linalg.norm(x-PointsArray[Index]) for x in PointsArray]
#    return PointsArray[np.where([IndexShift(PointsArray[Index],x) for x in PointsArray])]
    return PointsArray[np.argsort(d)[0:8]]
    
def normal_vector_v2(PointsArray,Index): #find normal by matrix invesion and setting z prefactor to 1
    Nf=Neighbours(PointsArray,Index)
    Centroid=PointsArray[Index]
    #find normal by matrix invesion and setting z prefactor to 1
    x,y,z=Nf[:,0]-Centroid[0],Nf[:,1]-Centroid[1],Nf[:,2]-Centroid[2]
    D=np.sum(x*x)*np.sum(y*y)-np.sum(x*y)**2
    if D!=0:#start by setting c=1
        a=np.sum(y*z)*np.sum(x*y)-np.sum(x*z)*np.sum(y*y)
        b=np.sum(x*y)*np.sum(x*z)-np.sum(x*x)*np.sum(y*z)        
        n=[a,b,D]
    else:#if that doesn't work, go with b=1
        D=np.sum(x*x)*np.sum(z*z)-np.sum(x*z)**2
        if D!=0:
            a=-np.sum(z*z)*np.sum(y*x)+np.sum(x*z)*np.sum(y*z)
            c=np.sum(x*z)*np.sum(y*z)-np.sum(x*x)*np.sum(y*z)
            n=[a,D,c]
        else:#if that doesn't work either, go with a=1
            D=np.sum(y*y)*np.sum(z*z)-np.sum(y*z)**2
            if D==0:
#                print("Could not performe plane interection.")
                return [0,0,0]
            b=np.sum(y*z)*np.sum(x*z)-np.sum(z*z)*np.sum(x*y)
            c=np.sum(y*z)*np.sum(x*z)-np.sum(y*y)*np.sum(x*z)
            n=[D,b,c]
          
    #One of the factors is multiplied by D, because that's how the others were computed
    #It's carried away with renormalisation anyway
    return n/np.linalg.norm(n)

def file_import(fname):
    r=list()
    with open(fname,'r') as csvfile:
        dataread=csv.reader(csvfile,delimiter=';')
        for row in dataread:
            r.append([float(row[i]) for i in range(len(row))])
    r=np.asarray(r)
    return r
def QuickPeek_H(C1,indices_1,C2,indices_2):
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=black)
    mlab.points3d(C1[:,0],C1[:,1],C1[:,2],color=white,scale_factor=1.2)
    mlab.points3d(C2[:,0],C2[:,1],C2[:,2],color=white,scale_factor=1.2)
    mlab.points3d(C1[indices_1][:,0],C1[indices_1][:,1],
                C1[indices_1][:,2],color=(1,0,0))
    mlab.points3d(C2[indices_2][:,0],C2[indices_2][:,1],
                C2[indices_2][:,2],color=(1,0,0))    
    img=mlab.screenshot()
    mlab.close()
    return img

def delaunay_edges(Centroids):
    tri = Delaunay(Centroids[:,1:4])
    tri=tri.simplices
    Contacts = list()
    for i in range(len(tri)):
        for j in range(len(tri[i])):
            a=sorted([tri[i,j],tri[i,(j+1)%len(tri[i])]])
            if a not in Contacts:
                Contacts.append(a)
    Contacts=np.asarray(Contacts)
    return Contacts
def file_writer(cfile,datas):
    with open(cfile,'w') as csvfile:
        datawriter=csv.writer(csvfile,delimiter=";")
        for i in range(len(datas)):
            datawriter.writerow(datas[i])
    csvfile.close()
def PairAngleWriting(TargetContacts,c_folder,co_file,ch_file):
    C1=ContourFinder(str(TargetContacts[0]),c_folder+co_file)
    C2=ContourFinder(str(TargetContacts[1]),c_folder+co_file)
    centroid_1,centroid_2=centroid(C1),centroid(C2)
    indices_1=points_in_cylinder(centroid_1,centroid_2,50,C1)
    indices_2=points_in_cylinder(centroid_2,centroid_1,50,C2)
    QuickPeek(C1,indices_1,C2,indices_2)
    x,y,x2,y2=Angles_In_A_Pair(TargetContacts,c_folder+co_file,c_folder+ch_file)
    return x,y,x2,y2
def ContactsDistances_centroids(ContactPairs,Centroids): #output distances for all centroid pairs
    d=[0]*len(ContactPairs)
    for i in range(len(ContactPairs)):
        cp_target=ContactPairs[i]
        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
        if int(v1[0])!=cp_target[0] and int(v2[0])!=cp_target[1]:
            print("Oh no")
        d[i]=np.sqrt((v1[1]-v2[1])**2+(v1[2]-v2[2])**2+(v1[3]-v2[3])**2)
    return d
def pairCorrelationFunction_3D(x, y, z, S, rMax, dr): #from ShockSolutions
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram

    # Find particles which are close enough to the cube center that a sphere of radius
    # rMax will not cross any face of the cube
    bools1 = x > rMax
    bools2 = x < (S - rMax)
    bools3 = y > rMax
    bools4 = y < (S - rMax)
    bools5 = z > rMax
    bools6 = z < (S - rMax)

    interior_indices, = where(bools1 * bools2 * bools3 * bools4 * bools5 * bools6)
    num_interior_particles = len(interior_indices)

    if num_interior_particles < 1:
        raise  RuntimeError ("No particles found for which a sphere of radius rMax\
                will lie entirely within a cube of side length S.  Decrease rMax\
                or increase the size of the cube.")

    edges = arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = zeros([num_interior_particles, num_increments])
    radii = zeros(num_increments)
    numberDensity = len(x) / S**3

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = sqrt((x[index] - x)**2 + (y[index] - y)**2 + (z[index] - z)**2)
        d[index] = 2 * rMax

        (result, bins) = histogram(d, bins=edges, normed=False)
        g[p,:] = result / numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = mean(g[:, i]) / (4.0 / 3.0 * pi * (rOuter**3 - rInner**3))

    return (g_average, radii, interior_indices)
def Delaunay_bonds(centroids):#get the edges from Delaunay triangulation, using centroids as seeds
    tri = Delaunay(centroids[:,1:4])
    tri=tri.simplices
    Contacts = list()
    for i in range(len(tri)):
        for j in range(len(tri[i])):
            a=sorted([tri[i,j],tri[i,(j+1)%len(tri[i])]])
            if a not in Contacts:
                Contacts.append(a)
    return np.asarray(Contacts)

def dist(r0,r1):#compute distance in 3D
    return ((r0[0]-r1[0])**2+(r0[1]-r1[1])**2+(r0[2]-r1[2])**2)**0.5

def double_radii_bonds(centroids,vol_file,offset=0.0):#compute bonds if distance from Delaunay edges is smaller than the sum of the radii
    #offset is a parameter to add to the measured radius, to compensate for erosion during image treatment
    vol=np.asarray(file_import(vol_file))
    rad=[(3.*x[1]/4./np.pi)**(1./3.) for x in vol]
    bonds_dr=list()
    bonds_D=Delaunay_bonds(centroids)
    for k in range(len(bonds_D)):
        v1,v2=bonds_D[k,0]+np.min(centroids[:,0]),bonds_D[k,1]+np.min(centroids[:,0])
        r1,r2=centroids[int(v1)][1:4],centroids[int(v2)][1:4]
        d=dist(r1,r2)
        if d<(offset+rad[int(v1)]+rad[int(v2)]):
            bonds_dr.append([bonds_D[k][0],bonds_D[k][1]])
    return bonds_dr
def sce_bonds(sce_file,offset=0):
    data=file_import(sce_file)
    distance=list()
    for i in range(len(data)):
        for j in range(i):
            distance.append([data[i][0],data[j][0],dist(data[i][1:4],data[j][1:4]),data[i][4]+data[j][4]])
    filtered=[x for x in distance if x[2]<(x[3]+offset)]
    return np.asarray([[int(x[0]),int(x[1])] for x in filtered])

def ContourWriter(Labels,fname):
    start=time.time()
    with open(fname, 'w') as csvfile:
        for number in sorted(np.unique(Labels))[1::]:
            for i in range(len(Labels)):
                if number in Labels[i]:
                    cnts = (sk.measure.find_contours(Labels[i]==number,level=0.99)[0]).astype(np.uint16)
                    spamwriter = csv.writer(csvfile, delimiter=';')
                    for x in cnts:
                        spamwriter.writerow([number,x[0],x[1],i])
    csvfile.close()
    print("Contour writing: "+elapsed_since(start))

def VolumesWriter(Labels,fname):
    start=time.time()
    target=sorted(np.unique(Labels))[1::]
    NV=[0]*len(target)
    for x in target:
        index=np.where(np.asarray(target)==x)[0][0]
        NV[index]=[x,np.shape(np.argwhere(Labels==x))[0]]
    with open(fname,'w') as csvfile:
        datawriter=csv.writer(csvfile,delimiter=';')
        for row in NV:
            datawriter.writerow(row)
    csvfile.close()  
    print("Volumes writing: "+elapsed_since(start))

def centroid_writing(contours,fname):#write the centroids of the contours in the file fname
    start=time.time()
    labels=np.unique(contours[:,0])
    with open(fname, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')
        for number in sorted(labels):
                pos=contours[contours[:,0]==number]                          
                spamwriter.writerow([number,np.mean(pos[:,1]),np.mean(pos[:,2]),np.mean(pos[:,3])])
    csvfile.close() 
    print("Centroid writing: "+elapsed_since(start))

def show_network_and_drops(targets,volumes,Centroids,ContactPairs,fname):
    targets=[x for x in targets if x in volumes[:,0]]
    mlab.figure(bgcolor=(0,0,0))
    step=100
    for x in targets:    
        C=ContourFinder(str(x),fname)
        C=np.asarray([y.astype(np.uint16) for y in C])
        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
        mlab.plot3d(C[:,0][0::step],C[:,1][0::step],C[:,2][0::step],color=droplet_color)   
    target_contacts=[x for x in ContactPairs+2 if x[0] in targets or x[1] in targets]
    for i in range(len(target_contacts)):
        cp_target=target_contacts[i]
        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
        if int(v1[0])!=cp_target[0] and int(v2[0])!=cp_target[1]:
            print("Oh no")
        mlab.plot3d([v1[1],v2[1]],[v1[2],v2[2]],[v1[3],v2[3]],
                            color=(1,0,0),tube_radius=2)
    mlab.show()



def show_network_and_drops_2(targets,volumes,Centroids,ContactPairs,fname):
    targets=[x for x in targets if x in volumes[:,0]]
    mlab.figure(bgcolor=(0,0,0))
    step=100
    for x in targets:    
        C=ContourFinder(str(x),fname)
        C=np.asarray([y.astype(np.uint16) for y in C])
        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
        mlab.plot3d(C[:,0][0::step],C[:,1][0::step],C[:,2][0::step],color=droplet_color)   
    target_contacts=[x for x in ContactPairs+2 if x[0] in targets and x[1] in targets]
    for i in range(len(target_contacts)):
        cp_target=target_contacts[i]
        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
        if int(v1[0])!=cp_target[0] and int(v2[0])!=cp_target[1]:
            print("Oh no")
        mlab.plot3d([v1[1],v2[1]],[v1[2],v2[2]],[v1[3],v2[3]],
                            color=(1,0,0),tube_radius=2)
    mlab.show()

def show_ndc(targets,volumes,Centroids,ContactPairs,fname,step=100):
    targets=[x for x in targets if x in volumes[:,0]]
    mlab.figure(bgcolor=(0,0,0))
    #step=100
    for x in targets:
#        print(x)
    	Cc = Centroids[Centroids[:,0]==x]
#    	print(Cc[0])
    	mlab.points3d(Cc[0,1],Cc[0,2],Cc[0,3],scale_factor=20)
    for x in targets:    
        C=ContourFinder(str(x),fname)
        C=np.asarray([y.astype(np.uint16) for y in C])
        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
        mlab.plot3d(C[:,0][0::step],C[:,1][0::step],C[:,2][0::step],color=droplet_color)   
    target_contacts=[x for x in ContactPairs if x[0] in targets and x[1] in targets]
    for i in range(len(target_contacts)):
        cp_target=target_contacts[i]
        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
        if int(v1[0])!=cp_target[0] and int(v2[0])!=cp_target[1]:
            print("Oh no")
        mlab.plot3d([v1[1],v2[1]],[v1[2],v2[2]],[v1[3],v2[3]],
                            color=(1,0,0),tube_radius=2)
    mlab.show()

def show_dc(targets,volumes,Centroids,fname,step=100):#only droplet contours and centroids
    targets=[x for x in targets if x in volumes[:,0]]
    mlab.figure(bgcolor=(0,0,0))
    #step=100
    for x in targets:    
        C=ContourFinder(str(x),fname)
        C=np.asarray([y.astype(np.uint16) for y in C])
        droplet_color=(np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))
        mlab.plot3d(C[:,0][0::step],C[:,1][0::step],C[:,2][0::step],color=droplet_color)
    for x in targets:
        C=Centroids[Centroids[:,0]==x]
        mlab.points3d(C[0],C[1],C[2],scale_factor=20)
#    target_contacts=[x for x in ContactPairs if x[0] in targets and x[1] in targets]
#    for i in range(len(target_contacts)):
#        cp_target=target_contacts[i]
#        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
#        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
#        if int(v1[0])!=cp_target[0] and int(v2[0])!=cp_target[1]:
#            print("Oh no")
#        mlab.plot3d([v1[1],v2[1]],[v1[2],v2[2]],[v1[3],v2[3]],
#                            color=(1,0,0),tube_radius=2)
    mlab.show()

    
def show_nc(targets,volumes,Centroids,ContactPairs,ccolor=(1,0,0)):
    targets=[x for x in targets if x in volumes[:,0]]
    mlab.figure(bgcolor=(0,0,0))
    for x in targets:
    	Cc = Centroids[Centroids[:,0]==x]
    	mlab.points3d(Cc[0,1],Cc[0,2],Cc[0,3],scale_factor=20) 
    target_contacts=[x for x in ContactPairs if x[0] in targets and x[1] in targets]
    for i in range(len(target_contacts)):
        cp_target=target_contacts[i]
        v1=Centroids[np.where(Centroids[:,0]==cp_target[0])][0]
        v2=Centroids[np.where(Centroids[:,0]==cp_target[1])][0]
        if int(v1[0])!=cp_target[0] and int(v2[0])!=cp_target[1]:
            print("Oh no")
        mlab.plot3d([v1[1],v2[1]],[v1[2],v2[2]],[v1[3],v2[3]],
                            color=ccolor,tube_radius=2)
    mlab.show()
    
    

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def bonds_threshold(pos,maxbondlength):
    tree=KDTree(pos,12)
    bonds = tree.query_pairs(maxbondlength, output_type='ndarray')
    return bonds

#this function writes down the s2s distances for ii-jj pair if they're not created yet
#caution : the s2s.exe file is not modular at all yet
def s2s_file_writing(ii,jj,Volumes,input_folder="./Mousse2/Mousse_2_contours/",output_folder="./Mousse2/s2s_tester/",cyrad=20):
    C1=ContourFinder(str(ii),input_folder+"contours.csv")
    C2=ContourFinder(str(jj),input_folder+"contours.csv")
    centroid_1,centroid_2=centroid(C1),centroid(C2)
    indices_1=points_in_cylinder(centroid_1,centroid_2,cyrad,C1)
    indices_2=points_in_cylinder(centroid_2,centroid_1,cyrad,C2)
    with open(output_folder+"cap_1.txt", 'w') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=' ')
        for i in range(len(indices_1)):                         
            spamwriter.writerow([int(C1[indices_1[i]][0]),
                int(C1[indices_1[i]][1]),int(C1[indices_1[i]][2])])
    csvfile.close() 
    with open(output_folder+"cap_2.txt", 'w') as csvfile:
        spamwriter = csv.writer(csvfile,delimiter=' ')
        for i in range(len(indices_2)):                          
            spamwriter.writerow([int(C2[indices_2[i]][0]),
                        int(C2[indices_2[i]][1]),int(C2[indices_2[i]][2])])
    csvfile.close() 
    s2s_file=output_folder+"s2s_"+str(ii)+"_"+str(jj)+".out"      
    header =[0]*5
    header[0]='#centroid_1 '+str(ii)+' '+str(centroid_1[0])+' '+str(centroid_1[1])+' '+str(centroid_1[2])+'\n'
    header[1]='#centroid_2 '+str(jj)+' '+str(centroid_2[0])+' '+str(centroid_2[1])+' '+str(centroid_2[2])+'\n'
    header[2]='#droplet_1-volume '+str(Volumes[Volumes[:,0]==ii][0,1])+'\n'
    header[3]='#droplet_2-volume '+str(Volumes[Volumes[:,0]==jj][0,1])+'\n'
    header[4]='#cylinder-radius '+str(cyrad)+'\n'
    file_o = open(s2s_file,'w')
    for i in range(len(header)):
        file_o.write(header[i])
    file_o.close()
#    print(s2s_file)
    os.system("./s2s.exe >> "+s2s_file)
