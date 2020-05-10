
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from keras.models import model_from_json
from PIL import Image
import cv2
from skimage.transform import resize

app = Flask(__name__)

# Model saved with Keras model.save()
def initmain():
    json_file = open('models/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("models/cancer_weights_best.hdf5")
    print("Loaded Model from disk")
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model,graph

def initbvsn():
    json_file = open('models/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("models/cancer_weights_bestbenign.hdf5")
    print("Loaded Model from disk")
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model,graph

def initinsvsinv():
    json_file = open('models/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("models/cancer_weights_bestcancerous.hdf5")
    print("Loaded Model from disk")
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model,graph

def cropimages(im):
    im=np.array(im)
    M=512
    N=512
    #print(im.shape)
    tiles=[im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    result=[resize(i, (350, 350)) for i in tiles]
    return result




# Check https://keras.io/applications/

print('Model loaded. Check http://127.0.0.1:5000/')

def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized
def Dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def my_PreProc(Data):
    assert(len(Data.shape)==4)
    assert (Data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(Data)
    #my preprocessing:
    train_imgs = Dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

def check_num(li):
    count0=li.count(0)
    count1=li.count(1)
    if(count0>count1):
        return 0
    else:
        return 1

def model_predict(img_path):
    #img = image.load_img(img_path, target_size=(2560,1920))
    im = Image.open(img_path)
    #im = im.resize((350,350))
    '''
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    '''
    im=np.array(im)
    Data=[]
    Data=cropimages(im)
    print(len(Data))
    Data=np.array(Data)
    Data=np.rollaxis(Data,3, 1) 
    Data=my_PreProc(Data)
    Data=np.moveaxis(Data,1, 3)
    output=[]
    output1=[]
    output2=[]
    #print(Data[0].shape)
    #out=model.predict(np.moveaxis(np.expand_dims(Data[1],axis=0),1,2))
    #print(out[0])
    c=0
    model,graph=initmain()
    for img in Data:
        out=model.predict(np.moveaxis(np.expand_dims(img,axis=0),1,2))
        #print(out)
        if(out[0][0]>0.5):
            output.append(1)
        else:
            output.append(0)
    res=check_num(output)
    if(res==0):
        model1,graph1=initbvsn()
        for img in Data:
            out1=model1.predict(np.moveaxis(np.expand_dims(img,axis=0),1,2))
            #print(out)
            if(out1[0][0]>0.5):
                output1.append(1)
            else:
                output1.append(0)
        if(check_num(output1)==1):
            return "Normal"
        else:
            return "Benign"
    else:    
        model2,graph2=initinsvsinv()
        for img in Data:
            out2=model2.predict(np.moveaxis(np.expand_dims(img,axis=0),1,2))
            #print(out)
            if(out2[0][0]>0.5):
                output2.append(1)
            else:
                output2.append(0)
        if(check_num(output2)==1):
            return "Invasive Carcinoma"
        else:
            return "InSitu Carcinoma"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(os.listdir(basepath))
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path)
        return result
    return None

#port = int(os.environ.get('PORT', 5000))
#app.run()



if __name__ == '__main__':
    app.run()
'''
# Serve the app with gevent
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
'''

