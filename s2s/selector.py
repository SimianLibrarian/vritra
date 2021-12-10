#this program will decide wether a pair (as described by its 2z2 datas) is a contact pair or not
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
import scipy.integrate as integrate
import os
from scipy.spatial import Delaunay
#import pyfiltering.bodycharacterisation as bdch
def Delaunay_bonds(centroids):
    tri = Delaunay(centroids[:,1:4])
    tri=tri.simplices
    Contacts = list()
    for i in range(len(tri)):
        for j in range(len(tri[i])):
            a=sorted([tri[i,j],tri[i,(j+1)%len(tri[i])]])
            if a not in Contacts:
                Contacts.append(a)
    return np.asarray(Contacts)
c_folder='../Mousse2/Mousse_2_contours/'
Centroids=bdch.file_import(c_folder+"centroids.csv")
bonds_D=Delaunay_bonds(Centroids)
#from mayavi import mlab
#3-2 : good, 19-16 : not good, but both look quite the same
#ii=3
#jj=2
#fname = "../Mousse2/s2s_tester/s2s_"+str(ii)+"_"+str(jj)+".out"
#t=np.genfromtxt(fname)
#histo = np.histogram(t,bins=50)
#x = [(histo[1][i]+histo[1][i+1])/2. for i in range(len(histo[1])-1)]
#y = histo[0]
#x,y=np.asarray(x),np.asarray(y)
#gauss = norm.fit(t)
#plt.plot(x,y)
#plt.plot(x,np.max(y)*np.exp(-((x-gauss[0])**2.)/2./gauss[1]**2))
#plt.show()
#%%First critertion : ad hoc distance threshold, with neighbours if a minimal number of distance are below it
d_thresh = 200

with open("../Mousse2/Mousse_2_contours/pairs_method1.csv","wb") as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=";")
    for i in range(len(bonds_D)):
        
        ii,jj=bonds_D[i]
        fname = "../Mousse2/s2s_tester/s2s_"+str(jj)+"_"+str(ii)+".out"
        if os.path.isfile(fname):
            print(str(i)+" over "+str(len(bonds_D)))
            t = np.genfromtxt(fname)
            ratio= np.sum(t<d_thresh)/float(len(t))
            if ratio >0.05:
                spamwriter.writerow([ii,jj])
csvfile.close()
#%%this is way too slow. First : make a threshold with simple delaunay bond length4
#import pyfiltering.bodycharacterisation as bdch
from scipy.spatial import Delaunay
def Delaunay_bonds(centroids):
    tri = Delaunay(centroids[:,1:4])
    tri=tri.simplices
    Contacts = list()
    for i in range(len(tri)):
        for j in range(len(tri[i])):
            a=sorted([tri[i,j],tri[i,(j+1)%len(tri[i])]])
            if a not in Contacts:
                Contacts.append(a)
    return np.asarray(Contacts)
c_folder='../Mousse2/Mousse_2_contours/'
Centroids=bdch.file_import(c_folder+"centroids.csv")
bonds_D=Delaunay_bonds(Centroids)
bonds_s2s = list()
d1=200
for i in range(len(bonds_D)):
    ii,jj=bonds_D[i]+1
    c1,c2=Centroids[Centroids[:,0]==ii][0,1::],Centroids[Centroids[:,0]==jj][0,1::]
    if ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**0.5<d1:
        bonds_s2s.append(bonds_D[i]+1)
bonds_s2s = np.asarray(bonds_s2s)

d_thresh=100
with open("../Mousse2/Mousse_2_contours/pairs_method1-bis.csv","wb") as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=";")
    for i in range(len(bonds_s2s)):
        print(str(i)+" over "+str(len(bonds_s2s)))
        ii,jj=bonds_s2s[i]
        fname = "../Mousse2/s2s_tester/s2s_"+str(jj)+"_"+str(ii)+".out"
        if os.path.isfile(fname):
            t = np.genfromtxt(fname)
            ratio= np.sum(t<d_thresh)/float(len(t))
            if ratio >0.05:
                spamwriter.writerow([ii,jj])
csvfile.close()

#with open("../Mousse2/Mousse_2_contours/pairs_method1-bis.csv","wb") as csvfile:
#    spamwriter=csv.writer(csvfile,delimiter=";")
#    for ii in range(len(bonds_s2s)):
#        for jj in range(1,ii):
#            fname = "../Mousse2/s2s_tester/s2s_"+str(ii)+"_"+str(jj)+".out"
#            if os.path.isfile(fname):
#                t=np.genfromtxt(fname) 
#                ratio= np.sum(t<d_thresh)/float(len(t))
#                if ratio >0.05:
#                    print(str(ii)+";"+str(jj))
#                    spamwriter.writerow([ii,jj])
#csvfile.close()
    
#%%Second workflow : threshold on the fitted distribution
bonds_D=Delaunay_bonds(Centroids)
bonds_s2s = list()
d1=200
for i in range(len(bonds_D)):
    ii,jj=bonds_D[i]+1
    c1,c2=Centroids[Centroids[:,0]==ii][0,1::],Centroids[Centroids[:,0]==jj][0,1::]
    if ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**0.5<d1:
        bonds_s2s.append(bonds_D[i]+1)
bonds_s2s = np.asarray(bonds_s2s)
d_thresh=150
#ii=9

#jj=6
#fname = "../Mousse2/s2s_tester/s2s_"+str(ii)+"_"+str(jj)+".out"
#t=np.genfromtxt(fname)
#gauss = norm.fit(t)
with open("../Mousse2/Mousse_2_contours/pairs_method2.csv","wb") as csvfile:
    spamwriter = csv.writer(csvfile,delimiter=";")
    with open("../Mousse2/Mousse_2_contours/pairs_method3.csv","wb") as csvfile2:
        spamwriter2 = csv.writer(csvfile2,delimiter=";")
        for i in range(len(bonds_s2s)):
            print(str(i)+" over "+str(len(bonds_s2s)))
            ii,jj=bonds_s2s[i]
            fname = "../Mousse2/s2s_tester/s2s_"+str(jj)+"_"+str(ii)+".out"
            if os.path.isfile(fname):
                t = np.genfromtxt(fname)                    
                gauss = norm.fit(t)                    
                cross,_ = integrate.quad(lambda x: np.exp(-((x-gauss[0])**2.)/2./gauss[1]**2),-np.inf,0)/np.sqrt(2.*np.pi)/gauss[1]
                Nthresh,_ = integrate.quad(lambda x: np.exp(-((x-gauss[0])**2.)/2./gauss[1]**2),-np.inf,d_thresh)/np.sqrt(2.*np.pi)/gauss[1]                    
                if cross>0.001:
                        spamwriter.writerow([ii,jj])
                if Nthresh>0.1:
#                    print(str(ii)+"-"+str(jj))
#                    print(gauss)
#                    print(cross,Nthresh)
                    spamwriter2.writerow([ii,jj])
csvfile.close()
csvfile2.close()
#%% Third critertion : ad hoc distance threshold, with neighbours if one distance is below threshold
d_thresh = 150
#this part works with the summarised.csv file (minimal s2s distance per pair)
t = np.genfromtxt('../Mousse2/s2s_tester/summarised.csv',delimiter=';')
tfiltered = t[t[:,2]<d_thresh]
with open("../Mousse2/Mousse_2_contours/pairs_method4.csv","wb") as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=';')
    for i in range(len(t)):
        spamwriter.writerow([int(t[i][0]),int(t[i][1])])
csvfile.close()

with open("../Mousse2/Mousse_2_contours/pairs_method4-bis.csv","wb") as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=';')
    for i in range(len(tfiltered)):
        spamwriter.writerow([int(tfiltered[i][0]),int(tfiltered[i][1])])
csvfile.close()
#This part works with the s2s files directly
#with open("../Mousse2/Mousse_2_contours/pairs_method4.csv","wb") as csvfile:
#    spamwriter=csv.writer(csvfile,delimiter=";")
#    for i in range(len(bonds_D)):
#        
#        ii,jj=bonds_D[i]+1
#        fname = "../Mousse2/s2s_tester/s2s_"+str(jj)+"_"+str(ii)+".out"
#        if os.path.isfile(fname):
#            print(str(i)+" over "+str(len(bonds_D)))
#            t = np.genfromtxt(fname)
#            d = np.min(t)
#            if d<d_thresh:
#                spamwriter.writerow([ii,jj])
#csvfile.close()
#%%Fourth criterion : first a large threshold on delaunay network
#then a second s2s
d_thresh=200
#targets=range(1,30)
c_folder = "./Mousse2/Mousse_2_contours/"
volumes=bdch.file_import(c_folder+"contours_volume.csv")
ContactPairs=bdch.file_import(c_folder+"pairs_method4.csv")
ContactPairsbis=bdch.file_import(c_folder+"pairs_method4-bis.csv")
Centroids=bdch.file_import(c_folder+"centroids.csv")
bonds_D=Delaunay_bonds(Centroids)
bonds_d=np.zeros(shape=(len(bonds_D),3))
for i in range(len(bonds_D)):
    c1,c2 = Centroids[Centroids[:,0]==bonds_D[i][0]+1][0],Centroids[Centroids[:,0]==bonds_D[i][1]+1][0]
    d = ((c1[1]-c2[1])**2+(c1[2]-c2[2])**2+(c1[3]-c2[3])**2)**0.5
    bonds_d[i]=[bonds_D[i][0],bonds_D[i][1],d]
bonds_d=np.asarray(bonds_d)
bonds_d=np.asarray([x for x in bonds_d if x[2]<d_thresh])
co_file="contours.csv"

d_thresh = 100
#this part works with the summarised.csv file (minimal s2s distance per pair)
t = np.genfromtxt('./Mousse2/s2s_tester/summarised.csv',delimiter=';')

with open("./Mousse2/Mousse_2_contours/pairs_method5.csv","wb") as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=';')
    for i in range(len(t)):
        spamwriter.writerow([int(t[i][0]),int(t[i][1])])
csvfile.close()

for i in range(len(t)):
    temp =[x for x in bonds_d[:,0:2]+1 if x[0]==t[i][0] and x[1]==t[i][1]]
    if len(temp)>0:
        print(temp)
        
        
#with open("./Mousse2/Mousse_2_contours/pairs_method5-bis.csv","wb") as csvfile:
#    spamwriter=csv.writer(csvfile,delimiter=';')
#    for i in range(len(tfiltered)):
#        spamwriter.writerow([int(tfiltered[i][0]),int(tfiltered[i][1])])
#csvfile.close()
#%%
#def QuickPeek():
#    black = (0,0,0)
#    white = (1,1,1)
#    mlab.figure(bgcolor=black)
#    mlab.points3d(C1[:,0],C1[:,1],C1[:,2],color=white,scale_factor=1.2)
#    mlab.points3d(C2[:,0],C2[:,1],C2[:,2],color=white,scale_factor=1.2)
#    mlab.show()
#    
#c_folder='../Mousse2/Mousse_2_contours/'
#co_file="contours.csv"
#C1=bdch.ContourFinder(str(ii),c_folder+co_file)
#C2=bdch.ContourFinder(str(jj),c_folder+co_file)
##centroid_1=bdch.centroid(C1)
##centroid_1=bdch.centroid(C2)
#QuickPeek()
##%%routine to get an histogram summarised 
#histo = np.histogram(t,bins=50)
##with open("../Mousse2/s2s_tester/summarised/s2s_summarised_"+str(ii)+"_"+str(jj)+".out","wb") as csvfile:
##    spamwriter=csv.writer(csvfile,delimiter=" ")
##    for kk in range(len(histo[0])):
##        spamwriter.writerow([histo[0][kk],histo[1][kk]])
##csvfile.close()
##%%
##thresh=0.3
#x = [(histo[1][i]+histo[1][i+1])/2. for i in range(len(histo[1])-1)]
#y = histo[0]
#x,y=np.asarray(x),np.asarray(y)
##plt.plot(x,y)
##plt.plot(x,[np.max(y)*thresh]*len(x),'-r')
##plt.fill_between(x[y<=np.max(y)*thresh],y[y<=np.max(y)*thresh],color='red')
##plt.fill_between(x[y>=np.max(y)*thresh],np.max(y)*thresh,color='orange')
##plt.show()
##suite = [y>=np.max(y)*thresh][0]
##s = list()
##
##    if i==0:
##        s.append(suite[i])
##    elif suite[i]!=s[-1]:
##        s.append(suite[i])
##s = np.asarray(s).astype(int)
##
##def array_identifier(a):
##    l = len(a)
##    if l==3:
##        if np.sum(a==np.asarray([0,1,0]))==3:
##            print("That's not a pair")
##        if np.sum(a==np.asarray([1,0]))==2:
##            print("That's a pair")
##    else:
##        print("I don't know what that is")
##array_identifier(s)
##%%
#from scipy.stats import norm
#gauss = norm.fit(t)
##mlab.normpdf(20,gauss[0],gauss[1])
#plt.plot(x,y)
#plt.plot(x,np.max(y)*np.exp(-((x-gauss[0])**2.)/2./gauss[1]**2))
#plt.show()
