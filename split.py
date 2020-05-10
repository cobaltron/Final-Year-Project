from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import cv2
import numpy as np

def cropimages(im):
    im=np.array(im)
    M=512
    N=512
    #print(im.shape)
    tiles=[im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    return tiles

data_path='BrCancer/benign/'
data1=os.listdir(data_path)

c=1
for i in range(0,len(data1)):
    img = load_img('BrCancer/benign/'+data1[i]) 
    tiles=cropimages(img)
    for i in tiles:
        cv2.imwrite('Dataset_1/Benign/img'+str(c)+'.png',i)
        c=c+1