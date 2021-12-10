import numpy as np
import scipy as scp
import scipy.stats
import cv2 as cv
import os
from PIL import Image
import skimage as sk
import skimage.restoration,skimage.morphology
import multiprocessing as mp
import progressbar


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
def PictureNormalisation(dat,m=0.5,M=99.5):
    vmin,vmax=scp.stats.scoreatpercentile(dat,(m,M))
    dat = np.clip(dat, vmin, vmax)
    dat = (dat - vmin) / (vmax - vmin)    
    return dat

def StackImportation(folder,N,step=1,skip=0):
    content=OrganiseSlices(folder)
    dat=list()
    im_frame=Image.open(folder+content[int(skip)])
    size=im_frame.size[1::-1]
    with progressbar.ProgressBar(max_value=N) as bar:
        k=0
        for i in range(N):
            bar.update(k)
            k=k+1
#			if i!=0:
            im_frame = Image.open(folder+content[int(skip)+i*int(step)])
#        dat.append(np.array(im_frame.getdata()).reshape(size).astype(int16))
            dat.append((np.array(im_frame.getdata()).astype(np.uint16)).reshape(size))
#        print("Table size: "+format_bytes(getsizeof(dat[i])))   
#    return dat
    return np.asarray(dat)  
#    content=OrganiseSlices(folder)
#    dat=list()
#    im_frame=Image.open(folder+content[int(skip)])
#    size=im_frame.size[1::-1]
#    for i in range(N):
#        if i!=0:
#            im_frame = Image.open(folder+content[int(skip)+i*int(step)])
#        dat.append(np.array(im_frame.getdata()).reshape(size).astype(np.int16))
##        print("Table size: "+format_bytes(getsizeof(dat[i])))   
##    return dat
#    return np.asarray(dat)

def StackImportation_resized(folder,N,step=1,skip=0,ratio=1.00):
    content=OrganiseSlices(folder)
    dat=[None]*N
    for i in range(N):
        im_frame = Image.open(folder+content[int(skip)+i*int(step)])
#        dat[i] = np.array(im_frame.getdata()).reshape(im_frame.size[1::-1])
        temp = np.array(im_frame.getdata()).reshape(im_frame.size[1::-1])
        width,height = int(temp.shape[1]*ratio),int(temp.shape[0]*ratio)
        dim = (width, height)
        temp = PictureNormalisation(temp,m=0,M=100)
        resized = cv.resize(temp, dim, interpolation = cv.INTER_AREA)
        dat[i]=resized
    return np.asarray(dat)

def BilateralFiltering(dat,intensity_kernel=0.1,spatial_kernel=4):
    bilateral=[None]*len(dat)
    for i in range(len(dat)):
        bilateral[i]=sk.restoration.denoise_bilateral(dat[i],sigma_color=intensity_kernel,sigma_spatial=spatial_kernel,multichannel=False)
    bilateral=np.asarray(bilateral)
    return dat 

def BilateralFiltering2D(dat,intensity_kernel=0.1,spatial_kernel=4):
    bilateral=[None]*len(dat)
    for i in range(len(dat)):
        bilateral[i]=sk.restoration.denoise_bilateral(dat[i],sigma_color=intensity_kernel,sigma_spatial=spatial_kernel,multichannel=False)
    bilateral=np.asarray(bilateral)
    return dat    

def collect_result(result):
    global results
    results.append(result)

def f(dat,index,sc,ss):
    return [index,sk.restoration.denoise_bilateral(dat,sigma_color=sc,sigma_spatial=ss,multichannel=False)]

def ParallelBilateralFiltering2D(dat,intensity_kernel=0.1,spatial_kernel=4):
    pool=mp.Pool(mp.cpu_count()-1)
    global results
    results =[]
    for i in range(len(dat)):
        pool.apply_async(f, args=(dat[i],i,intensity_kernel,spatial_kernel), callback=collect_result)
    #    pool.apply(f, args=(dat[i],0.1,4), callback=collect_result)
    pool.close()
    pool.join()
    return results

def Sorting(out):
    indices = [x[0] for x in out]
    out_f = [out[i][1] for i in np.argsort(indices) ]
    return out_f

def StackSave(data,i_folder,o_folder):
    content=OrganiseSlices(i_folder)
    for i in range(len(data)):
        result = Image.fromarray(data[i])
        result.save(o_folder+'bilateral_'+str(content[i]))


    
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

def StackRadialProfile(dat,center):
    r=[None]*len(dat)
    for i in range(len(dat)):
        r[i]=radial_profile(dat[i],center)
    r=np.asarray(r)
    
    return [np.mean(r[:,i]) for i in range(len(r[0]))]

def RadialCorrection(img,data,center,r_profil):
    
    y,x=np.indices((img.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
#    plt.imshow(r),plt.show()
#    plt.plot(r_profil),plt.show()
    for i in range(len(r)):
        for j in range(len(r[i])):
            if type(r[i,j])!=np.int64:
                print(r[i,j])
            else:
                img[i,j]=img[i,j]/r_profil[r[i,j]]
    return img
def StackRadialCorrection(data):
    center=[len(data[0])/2,len(data[0][0])/2]
    r_profil=StackRadialProfile(data,center)
    for i in range(len(data)):
        RadialCorrection(data[i],data,[len(data[i])/2,len(data[i][0])/2],r_profil)
    return data
        
def StackCropping(img,threshold):
    sample=img>threshold
    sample=scp.ndimage.binary_fill_holes(sample)
    #sample=sk.morphology.opening(sample,sk.morphology.ball(2))
    sample=sk.morphology.opening(sample,sk.morphology.cube(12))
#    plt.imshow(sample[0])
    sample=sk.morphology.closing(sample,sk.morphology.cube(4))
    bbox = scp.ndimage.find_objects(sample)
    mask = sample[bbox[0]]
#    plt.imshow(mask[0])
    img=img[bbox[0]]    
    return img,bbox,mask

def StackSegmentation(img,cropping='yes',mask='no',threshold=0.4,imin=0.4,imax=0.7):
    if cropping=='yes':
        if mask=="no":
            img,bbox,mask=StackCropping(img,threshold)
        else:
            img,bbox=StackCropping(img,threshold)[0:2]
    else:
        if mask=="no":
            mask=StackCropping(img,threshold)[2]
#        else:
#            bbox=StackCropping(img,threshold)[1]        
    bi_copy=np.copy(img)
    bi_copy[bi_copy>imax]=imin/200
    bi_copy= bi_copy>imin
    bi_copy=scp.ndimage.binary_fill_holes(bi_copy)
    bi_copy=sk.morphology.binary_opening(bi_copy,sk.morphology.cube(2))
    bi_copy=bi_copy*mask[0]
    bi_copy=scp.ndimage.binary_fill_holes(bi_copy)
    bi_copy=sk.morphology.opening(bi_copy,sk.morphology.cube(4))
    bi_copy=sk.morphology.closing(bi_copy,sk.morphology.cube(4))
#    bi_copy=sk.morphology.remove_small_objects(bi_copy,min_size=20)
    binarised=np.copy(bi_copy)
    binarised=binarised+1
    rw=sk.segmentation.random_walker(img,binarised,beta=10,mode='cg')
    return rw

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

def Labels_SingleSlice(l,t,fname="./DatasICS/SliceYDroplets10um/Volumes.txt",wmod="a"):       
    with open(fname,mode=wmod) as o_file:
        if wmod=="w":
            o_file.write("# Width "+str(min(l.shape)) +" Length "+str(max(l.shape))+"\n")
        o_file.write("#Label x y z \n")
        o_file.write("# New slice "+str(t)+"\n")
        for i in range(len(l)):
            for j in range(len(l[i])):
                if l[i][j]!=1:
#                    o_file.write(str(l[i][j])+" "+str(i)+" "+str(j)+"\n")
                    o_file.write(str(l[i][j])+" "+str(i)+" "+str(j)+" "+str(t)+"\n")
    o_file.close()

def Labels(labels,fname="./DatasICS/SliceYDroplets10um/Volumes.txt"):
    for i in range(len(labels)):
        print(i)
        if i==0:
            Labels_SingleSlice(labels[i],i,fname=fname,wmod="w")
        else:
            Labels_SingleSlice(labels[i],i,fname=fname,wmod="a")

def ImageSaving(bilateral,o_folder,N_img,step,header,intensity_kernel,spatial_kernel):
#    o_folder = '.'+i_folder[1:i_folder[0:-1].rfind('/')+1]
    c = os.listdir(o_folder)
#    if 'SlicesY_BilateralFiltered' not in c:
#        os.mkdir('.'+i_folder[1:i_folder[0:-1].rfind('/')+1]+'SlicesY_BilateralFiltered/')
#    o_folder = '.'+i_folder[1:i_folder[0:-1].rfind('/')+1]+'SlicesY_BilateralFiltered/'
    for i in range(len(bilateral)):
        index = str(i)
        while len(index)<4:
            index='0'+index
        cv.imwrite(o_folder+'slice'+str(header+int(index))+'.tiff',bilateral[i])
    f = open(o_folder+'FilterParameters.txt','w')
    f.write('N_images Step Header IntensityKernel SpatialKernel\n')
    f.write(str(N_img)+' '+str(step)+' '+str(header)
    +' '+str(intensity_kernel)+' '+str(spatial_kernel))
    f.close()
    
def ReadFiltered(o_folder):
    c = os.listdir(o_folder)
    c = [x for x in c if '.tiff' in x]
    dat = [None]*len(c)
    for i in range(len(dat)):
        dat[i] = cv.imread(o_folder+c[i],cv.IMREAD_ANYDEPTH)
    return dat

def Is_AlreadyFiltered(i_folder,N_img,step,header,intensity_kernel,spatial_kernel):
    fname='SlicesY_BilateralFiltered'
    o_folder = '.'+i_folder[1:i_folder[0:-1].rfind('/')+1]   
    c = os.listdir(o_folder)
    if fname in c:
        c2 = os.listdir(o_folder+fname)
        if 'FilterParameters.txt' in c2:
            t = np.genfromtxt(o_folder+fname+'/FilterParameters.txt',dtype='str')
            t=list(t)
            t[0],t[1]=list(t[0]),list(t[1])
            n_file = float(t[1][t[0].index('N_images')])
            step_file = float(t[1][t[0].index('Step')])
            header_file = float(t[1][t[0].index('Header')])
            si_file = float(t[1][t[0].index('IntensityKernel')])
            ss_file = float(t[1][t[0].index('SpatialKernel')])
            if n_file == N_img and step_file==step and header_file==header and si_file==intensity_kernel and ss_file == spatial_kernel:
                print("Images with these parameters have already been produced.\n"
                      +"Switching to image import instead.")
                return True
    return False
                
