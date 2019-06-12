import numpy as np

class VolumeReconstructor:
    def __init__(self):
        pass
# The first step should be to identify the number of circles one can get from a picture
# 1- Get how many circles N are possible to identify : repetition over x slices
# 2- Declare N arrays, which will be incrementally concatenated
# 3- Compute each array/volume separately

    def counting(self,fname):
        labels=np.asarray(np.genfromtxt(fname,usecols=0),dtype=int)
        t=np.asarray(np.genfromtxt(fname,usecols=(1,2,3)))     
        with open(fname) as f:
            for line in f.readlines():
                li = line.lstrip()
                if li.startswith("#"):
                                 i1,i2=1,li.rfind(" ")
        l=float(''.join( [li[i1+j] for j in range(i2-i1) if li[i1+j].isdigit()==True] ))
        L=float(''.join([li[i2+j] for j in range(len(li)-i2) if li[i2+j].isdigit()==True]))
        t[:,0],t[:,1],t[:,2],N=t[:,0]/l,t[:,1]/L,t[:,2]/L,0
        indices=list()
        for k in range(len(t)):
            a=[i-1 for i in range(k+1,len(t)) if np.allclose(t[k][0:2],t[i-1][0:2],rtol=5e-2)==True]
            if len(a)>1 and np.any( [a[i] in indices[j] for j in range(len(indices)) for i in range(len(a))  ] )==False:
                indices.append(a)           
                N=N+1
#        ind_bis=indices
        for j in range(len(indices)):
            if j>=len(indices):
                break
            p=labels[indices[j]]
            if np.all(np.linspace(min(p),max(p),num=len(p))==sorted(p))!=True:
                try:
                    del indices[j]
                    j=j-1
                except:
                    print("Error")
#        indices=ind_bis
        return N,indices
                #np.any( [[0,2][i] in indices[j] for j in range(len(indices)) for i in range(2)  ] )
#        q=np.concatenate([[labels[0]],t[0]])
#        for i in range(len(a)):
#            q0=np.concatenate([[labels[a[i]]],t[a[i]]])
#            q=[q,q0]
    def expanding(self,fname):
        N,indices=counting(self,fname)
        
        
    def slicing(self,fname):
        #get the infos from the file
        #if 2 circles in 2 =/= slices have center different from less than x %, the 2 circles are bound together
        labels=np.asarray(np.genfromtxt(fname,usecols=0))
        t=np.asarray(np.genfromtxt(fname,usecols=(1,2)))
        #first step is to rescale all dimensions so as to better understand the position variations
        with open(fname) as f:
            for line in f.readlines():
                li = line.lstrip()
                if li.startswith("#"):
                                 i1,i2=1,li.rfind(" ")
        l=float(''.join( [li[i1+j] for j in range(i2-i1) if li[i1+j].isdigit()==True] ))
        L=float(''.join([li[i2+j] for j in range(len(li)-i2) if li[i2+j].isdigit()==True]))
        t[:,0],t[:,1]=t[:,0]/l,t[:,1]/L
        for i in range(len(np.unique(labels))):
            args2=[j for j in range(len(labels)) if labels[j]==np.unique(labels)[i]] 
            args1=[j for j in range(len(labels)) if abs(labels[j]-np.unique(labels)[i])==1]
#            if (abs(t[args2]-t[args1])<=0.01)==[True,True]:
#                print("Yeah")
        temp=[t[args2][i] for i in range(len(t[args2])) if np.all(abs(t[args2][i]-t[args1])<=0.01)]
        r=t[args1]
        if len(temp)!=0:
            r=np.concatenate([r,temp])
        return r