import numpy as np
import os
import cv2



#determining the image size ; firstby ll=ooking in the first 10 lines of the data file, and if not mention, will import one image to look at its size
#caution : this will lead to errors with resized pictures
def ImageSize(infile):
    f = open(infile,'r')
    for i in range(10):
        line = f.readline()
        if "#" in line[0]:
            if "width" in line or "Width" in line or "length" in line or "Length" in line:
#                print(line)
                line=line[0:-1].split(" ")
                f.close()
                if "width" in line:
                    i_w = line.index("width")+1
                elif "Width" in line:
                    i_w = line.index("Width")+1
                else:
                    print("No width mentionned.")
                
                if "length" in line:
                    i_l = line.index("length")+1
                elif "Length" in line:
                    i_l = line.index("Length")+1
                else:
                    print("No length mentioned")    
                
                l,w = float(line[i_l]),float(line[i_w])
                break
            else:
                print("No dimensions mentioned. Will try to import one slice to get its (x,y) dimensions.")
                c = os.listdir(infile[0:infile.rfind("/")+1])
                c = [w for w in c if '.' not in w]
                for i in range(len(c)):
                    c2 = os.listdir(infile[0:infile.rfind("/")+1]+c[i]+'/')
                    j = [w for w in range(len(c2)) if '.tif' in c2[w]]
                    if len(j)>1:
                        img = cv2.imread(infile[0:infile.rfind("/")+1]+c[i]+'/'+c2[j[0]] )
                        l,w = max(shape(img)[0:2]),min(shape(img)[0:2])
                        break
    return int(l),int(w)
#%%
def Labels_FromFile(infile):
    dat = np.genfromtxt(infile)
    la,x,y,z = dat[:,0],dat[:,1],dat[:,2],dat[:,3]
    la = [int(w) for w in la]
    x = [int(w) for w in x]
    y = [int(w) for w in y]
    z = [int(w) for w in z]
    thickness = len(np.unique(z))
    z_phantom = sorted(np.unique(z))
    length,width = ImageSize(infile)
    labels_i = np.ones(shape=(thickness,length,width ),dtype='int' )    
    for i in range(len(x)):
        labels_i[np.where(z[i] in z_phantom)[0],x[i],y[i]]=la[i]
    return labels_i