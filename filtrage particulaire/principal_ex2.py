import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.image as mpimg
from PIL import Image
import math
from scipy.stats import norm
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import os

def multinomial_resample(weights):

    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))


def lecture_image() :

    SEQUENCE = "./sequences/sequence1/"
    #charge le nom des images de la séquence
    filenames = os.listdir(SEQUENCE)
    T = len(filenames)
    #charge la premiere image dans ’im’
    tt = 0

    im=Image.open((str(SEQUENCE)+str(filenames[tt])))
    plt.imshow(im)
    
    return(im,filenames,T,SEQUENCE)

def selectionner_zone() :

    #lecture_image()
    print('Cliquer 4 points dans l image pour definir la zone a suivre.') ;
    zone = np.zeros([2,4])
 #   print(zone))
    compteur=0
    while(compteur != 4):
        res = plt.ginput(1)
        a=res[0]
        #print(type(a)))
        zone[0,compteur] = a[0]
        zone[1,compteur] = a[1]   
        plt.plot(a[0],a[1],marker='X',color='red') 
        compteur = compteur+1 

    #print(zone)
    newzone = np.zeros([2,4])
    newzone[0, :] = np.sort(zone[0, :]) 
    newzone[1, :] = np.sort(zone[1, :])
    
    zoneAT = np.zeros([4])
    zoneAT[0] = newzone[0,0]
    zoneAT[1] = newzone[1,0]
    zoneAT[2] = newzone[0,3]-newzone[0,0] 
    zoneAT[3] = newzone[1,3]-newzone[1,0] 
    #affichage du rectangle
    #print(zoneAT)
    xy=(zoneAT[0],zoneAT[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None') 
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.show(block=False)
    return(zoneAT)


def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de référence
    
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
    print(image_array_sample.shape)
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb
            
    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)

def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image



def calcul_histogramme(im,zoneAT,Nb):

  #  print(zoneAT)
    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])
   # print(box)
    littleim = im.crop(box)
##    plt.imshow(littleim)
##    plt.show()
    new_im,kmeans= rgb2ind(littleim,Nb)
    histogramme=np.asarray(new_im.histogram())
##  print(histogramme)
    histogramme=histogramme/np.sum(histogramme)
  #  print(new_im)
    return (new_im,kmeans,histogramme)




N=100
Nb=20
ecart_type=np.sqrt(50)
lambda_im=60
c1=900
c2=900
C=np.diag([c1,c2])  

[im,filenames,T,SEQUENCE]=lecture_image()   
zoneAT=selectionner_zone()
new_im, kmeans, histo_ref=calcul_histogramme(im,zoneAT,Nb)
print(zoneAT)


particules=np.array([zoneAT[0],zoneAT[1]]) + np.dot(np.random.randn(N,2),np.linalg.cholesky(C))
plt.plot(particules[:,0],particules[:,1],"X",color='blue')
poids=np.ones(N)/N
for t in range(1,T):
    print(t)
    plt.close()
    im=Image.open((str(SEQUENCE)+str(filenames[t])))
    
    #plt.show()
    #plt.pause(0.1)
   

    for j in range(0,N):
        particules[j,:]=particules[j,:] + np.dot(np.random.randn(1,2),np.linalg.cholesky(C))
        zoneAT_particule=[particules[j,0],particules[j,1],zoneAT[2],zoneAT[3]]
        
##        box=(particules[j,0],particules[j,1],particules[j,0]+zoneAT[2],particules[j,1]+zoneAT[3])
##        littleim = im.crop(box)
##        new_im,kmeans_bis= rgb2ind(littleim,kmeans)
##        histo_particules=np.asarray(new_im.histogram())
##        histo_particules=histo_particules[0:256]/np.sum(histo_particules[0:256])
        [new_im_bis,kmeans_bis,histo_particules]= calcul_histogramme(im,zoneAT_particule,kmeans)
        Distance=np.sqrt(1-np.sum(np.sqrt(histo_particules*histo_ref)))
        poids[j]=np.exp(-lambda_im*(Distance**2))
        
    poids=poids/np.sum(poids)
    estimateur=np.dot(particules.transpose(),poids)
    neff=1/sum(poids**2)
    print(neff)
    index=multinomial_resample(poids)
    poids=np.ones(N)/N
    particules=particules[index,:]
    plt.imshow(im)
    plt.plot(particules[:,0],particules[:,1],"X",color='blue')
    xy=(estimateur[0],estimateur[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None') 
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.pause(0.5)
    

    
    


    
