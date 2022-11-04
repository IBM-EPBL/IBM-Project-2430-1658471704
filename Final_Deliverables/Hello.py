import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
import numpy as np
import cv2
from skimage import feature
import os
from flask import Flask, redirect, url_for, request , render_template
from werkzeug.utils import secure_filename
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'static/uploads' 

#le = LabelEncoder()
model = pickle.load(open('parkinsons.pkl','rb'))

def quantify_image(image):
    # compute the histogram 
    features = feature.hog(image, orientations=9,
    pixels_per_cell=(10, 10), cells_per_block=(2, 2),
    transform_sqrt=True, block_norm="L1")
    # return the feature vector
    return features

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/getDetails',methods = ['POST'])
def getDetails():
    img=request.files['Image']
    name=request.form['Name']
    age=request.form['Age']
    print(img.filename)
    filename = secure_filename(img.filename)
    img.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    print(os.path.join(app.config['UPLOAD_PATH'], filename))
    image = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], filename))
    output = image.copy()
    output = cv2.resize(output , (128,128))
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image , (200,200))
    image = cv2.threshold(image , 0 , 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    features = quantify_image(image)
    print(features)
    preds = model.predict([features])
    print(preds)
    if preds[0]==1:
        return render_template('predictionResult.html',Name=name,Age=age,Result=" Parkinson Detected ",imageName= filename)
    else:
        return render_template('predictionResult.html',Name=name,Age=age,Result=" Healthy Person ",imageName= filename)
if __name__ == '__main__':
   app.run("127.0.0.1",5000)