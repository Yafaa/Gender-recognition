#import section
from flask import Flask , render_template, request,abort, current_app, make_response
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import skimage
import skimage.io
from skimage import transform
from mimetypes import guess_extension
import json
import PIL
import matplotlib.pyplot as plt
import librosa.display
import joblib  
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pickle
from skimage.transform import resize
import pywt
import config 

#define the app
app = Flask(__name__)

# Attributes 
hop_length = 512 # number of samples per time-step in spectrogram
n_mels = 128 # number of bins in spectrogram. Height of image
time_steps = 384 # number of time-steps. Width of image
# extract a fixed length window
start_sample = 0 # starting at beginning
length_samples = time_steps*hop_length
img_height=128
img_width=256

#Classes
class_names=['Female', 'Male']


    
@app.route("/spec")
def Index():
    return render_template(config.home)

@app.route("/")
def home():
    return render_template(config.index)

@app.route("/record")
def record():
    return render_template(config.record)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def Process(file):
  y, sr = librosa.load(file)
  

  window = y[start_sample:start_sample+length_samples]
  # use log-melspectrogram
  mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=hop_length*2, hop_length=hop_length)
  mels = np.log(mels + 1e-9) # add small number to avoid log(0)

  # min-max scale to fit inside 8-bit range
  img = scale_minmax(mels, 0, 255).astype(np.uint8)
  img = np.flip(img, axis=0) # put low frequencies at the bottom in image
  img = 255-img # invert. make black==more energy   
  # save as PNG
  #skimage.io.imsave(out, img)

  return img





def Model(file_path):
  
    modelA = keras.models.load_model(config.modelAll)
    model2 = keras.models.load_model(config.CNN2)
    model3 = keras.models.load_model(config.CNN3)
    model4 = keras.models.load_model(config.CNN4)
        # settings
   
    imgName=file_path+".png"
    img = Process(file_path) 
    skimage.io.imsave(imgName, img)
    img = keras.preprocessing.image.load_img(
      imgName, target_size=(img_height, img_width))
    #img = transform.resize(img,(img_height,img_width))
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0) 
    #img=tf.image.grayscale_to_rgb(img)
    #predictions =( modelA.predict(img))
 

    predictions =( modelA.predict(img)  +  model2.predict(img)  +   model3.predict(img)   +   model4.predict(img))
    score = tf.nn.softmax( predictions[0])
    className ="This sound most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    print(className)
    return className

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method =='POST':
        print(request.form.get('Name'))
        print(request.form.get('Gender'))
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
      
        try:
            name=str(request.form['Name'])
            
            pred=Model(file_path)
            model_prediction =  "Hello "+ str(name) + "! Welcome to our website!  Our Algo predicted a  "+str(pred) +" gender!"
            return render_template(config.predict,prediction=model_prediction)
            
        except ValueError:
            return "Please Enter valid values"

@app.route("/rec")
def Rec():
    return render_template(config.Rec)        


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' in request.files:
        file = request.files['audio_file']
        # Get the file suffix based on the mime type.
        extname = guess_extension(file.mimetype)
        if not extname:
            abort(400)

        # Test here for allowed file extensions.

        # Generate a unique file name with the help of consecutive numbering.
        i = 1
        basepath = os.path.dirname(__file__)
        while True:
            dst = os.path.join(
                basepath ,
                'testUpload',
                secure_filename(f'audio_record_{i}{extname}'))
            if not os.path.exists(dst): break
            i += 1

        # Save the file to disk.
        file.save(dst)
        pred=Model(dst)
        model_prediction =  "Hello ! " + " Welcome to our website!  the CNN  Melscptogram  Algo predicted a  "+str(pred) +" gender!"
        
        return json.dumps( model_prediction) 
        #return make_response("", 200)
    
    abort(400)

@app.route("/rec2")
def Rec2():
    return render_template(config.Rec)       

@app.route('/upload2', methods=['POST'])
def upload2():
    if 'audio_file' in request.files:
        file = request.files['audio_file']
        # Get the file suffix based on the mime type.
        extname = guess_extension(file.mimetype)
        if not extname:
            abort(400)

        # Test here for allowed file extensions.

        # Generate a unique file name with the help of consecutive numbering.
        i = 1
        basepath = os.path.dirname(__file__)
        while True:
            dst = os.path.join(
                basepath ,
                'testUpload',
                secure_filename(f'audio_record_{i}{extname}'))
            if not os.path.exists(dst): break
            i += 1

        # Save the file to disk.
        file.save(dst)
        pred=ModelChorm(dst)
        model_prediction =  "Hello ! " + " Welcome to our website!  Our  chorma and cqt Algo predicted a  "+str(pred) +" gender!"
        
        return json.dumps( model_prediction) 
        #return make_response("", 200)
    
    abort(400)


################################################################################################################# CQT

def ProcessCQT(file):
  y, sr = librosa.load(file)
  window = y[start_sample:start_sample+length_samples]
  cqt =np.abs(librosa.cqt(y, sr=sr))
  # min-max scale to fit 
  img = scale_minmax(cqt , 0, 255)
  
  
  return img

def ModelCQT(file_path):
  
    model = keras.models.load_model(config.CQT)

    # settings
    img_height = 150
    img_width = 240
    imgName=file_path+".png"
    img = ProcessCQT(file_path) 
    skimage.io.imsave(imgName, img)
    img = keras.preprocessing.image.load_img(
      imgName, target_size=(img_height, img_width))
    
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0) 
    
 

    predictions =( model.predict(img))
    score = tf.nn.softmax( predictions[0])
    className ="This sound most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    print(className)
    return className

@app.route("/recQ")
def rec3():
    return render_template(config.recQ)       

@app.route('/upload3', methods=['POST'])
def upload3():
    if 'audio_file' in request.files:
        file = request.files['audio_file']
        # Get the file suffix based on the mime type.
        extname = guess_extension(file.mimetype)
        if not extname:
            abort(400)

        # Test here for allowed file extensions.

        # Generate a unique file name with the help of consecutive numbering.
        i = 1
        basepath = os.path.dirname(__file__)
        while True:
            dst = os.path.join(
                basepath ,
                'testUpload',
                secure_filename(f'audio_record_{i}{extname}'))
            if not os.path.exists(dst): break
            i += 1

        # Save the file to disk.
        file.save(dst)
        pred=ModelCQT(dst)
        model_prediction =  "Hello ! " + " Welcome to our website!  Our  CQT Algo predicted a  "+str(pred) +" gender!"
        
        return json.dumps(model_prediction) 
       
    
################################################################################################################# Chroma

def ProcessC(file):
  y, sr = librosa.load(file)
  window = y[start_sample:start_sample+length_samples]
  cqt =np.abs(librosa.cqt(y, sr=sr))
  # min-max scale to fit 
  img = scale_minmax(cqt , 0, 255)
  
  return img

def ModelC(file_path):
  
    model = keras.models.load_model(config.chroma)

    # settings
    img_height = 150
    img_width = 240
    imgName=file_path+".png"
    img = ProcessC(file_path) 
    skimage.io.imsave(imgName, img)
    img = keras.preprocessing.image.load_img(
      imgName, target_size=(img_height, img_width))
    
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0) 
    
 

    predictions =( model.predict(img))
    score = tf.nn.softmax( predictions[0])
    className ="This sound most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    print(className)
    return className

@app.route("/recC")
def rec4():
    return render_template(config.recC)       

@app.route('/upload4', methods=['POST'])
def upload4():
    if 'audio_file' in request.files:
        file = request.files['audio_file']
        # Get the file suffix based on the mime type.
        extname = guess_extension(file.mimetype)
        if not extname:
            abort(400)

        # Test here for allowed file extensions.

        # Generate a unique file name with the help of consecutive numbering.
        i = 1
        basepath = os.path.dirname(__file__)
        while True:
            dst = os.path.join(
                basepath ,
                'testUpload',
                secure_filename(f'audio_record_{i}{extname}'))
            if not os.path.exists(dst): break
            i += 1

        # Save the file to disk.
        file.save(dst)
        pred=ModelC(dst)
        model_prediction =  "Hello ! " + " Welcome to our website!  Our  Chromoa Algo predicted a  "+str(pred) +" gender!"
        
        return json.dumps(model_prediction) 


########################################################################################### Wavelet

def ProcessW(file):
  y, sr = librosa.load(file)
  window = y[start_sample:start_sample+length_samples]
  cqt =np.abs(librosa.cqt(y, sr=sr))
  # min-max scale to fit 
  img = scale_minmax(cqt , 0, 255)
  
  return img

  # amount of pixels in X and Y 
  rescale_size = 62
  # determine the max scale size
  n_scales = 62

  y, sr = librosa.load(file)

  scales = (1, len(y))
  #coeffs, freqs =pywt.cwt(y, scales =scales,wavelet='morl',method='fft')
  #coeffs, freqs =pywt.cwt(y, scales =scales,wavelet='morl',method='conv')
  coeffs, freqs =pywt.cwt(y, scales =scales,wavelet='morl',method='fft')         
  # resize the 2D cwt coeffs
  rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
  # save as PNG
  skimage.io.imsave(out, rescale_coeffs)
    
  return img

def ModelW(file_path):
  
    model = keras.models.load_model(config.wavelet)

    # settings
    img_height = 62
    img_width = 62
    imgName=file_path+".png"
    img = ProcessW(file_path) 
    skimage.io.imsave(imgName, img)
    img = keras.preprocessing.image.load_img(
      imgName, target_size=(img_height, img_width))
    
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0) 
    
 

    predictions =( model.predict(img))
    score = tf.nn.softmax( predictions[0])
    className ="This sound most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    print(className)
    return className

@app.route("/recW")
def rec5():
    return render_template(config.recW)       

@app.route('/upload5', methods=['POST'])
def upload5():
    if 'audio_file' in request.files:
        file = request.files['audio_file']
        # Get the file suffix based on the mime type.
        extname = guess_extension(file.mimetype)
        if not extname:
            abort(400)

        # Test here for allowed file extensions.

        # Generate a unique file name with the help of consecutive numbering.
        i = 1
        basepath = os.path.dirname(__file__)
        while True:
            dst = os.path.join(
                basepath ,
                'testUpload',
                secure_filename(f'audio_record_{i}{extname}'))
            if not os.path.exists(dst): break
            i += 1

        # Save the file to disk.
        file.save(dst)
        pred=ModelC(dst)
        model_prediction =  "Hello ! " + " Welcome to our website!  Our  Morlet wavelet Algo predicted a  "+str(pred) +" gender!"
        
        return json.dumps(model_prediction) 






if __name__ == '__main__':
    app.run(host=config.host,port=config.port)
    #app.run()