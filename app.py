from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import tensorflow as tf


app = Flask(__name__)


model = load_model('model_vgg16.h5')
model.make_predict_function()
def predict_label(img_path):
    
    new_images = []
    temp = np.array(tf.keras.preprocessing.image.load_img(img_path, color_mode = "rgb", 
                                                          target_size = (224, 224), 
                                                          interpolation="lanczos")) / 255.0
    new_images.append(temp)
    new_images = np.array(new_images)
    predictions = model.predict(new_images)
    pred_ints = predictions.argmax(axis = -1)
    
    ch=''
    if pred_ints[0]==0:
        ch=' is Bird'
    elif pred_ints[0]==1 :
        ch='is Cat'
    elif pred_ints[0]==2:
        ch='is Chicken'
    elif pred_ints[0]==3:
        ch='is Cow'
    elif pred_ints[0]==4:
        ch='is Dog'
    elif pred_ints[0]==5:
        ch='is Elephant'
    elif pred_ints[0]==6:
        ch='is Fish'
    elif pred_ints[0]==7:
        ch='is Fox'
    elif pred_ints[0]==8:
        ch='is Frog'
    elif pred_ints[0]==9:
        ch='is Giraffe'
    elif pred_ints[0]==10:
        ch='is Horse'
    elif pred_ints[0]==11:
        ch='is Lion'
    elif pred_ints[0]==12:
        ch='is Monkey'
    elif pred_ints[0]==13:
        ch='is Sheep'
    else:
        ch='no result'
    return(ch)


@app.route("/", methods = ['GET', 'POST'])
def main():
    return render_template("index.html",x=False)

@app.route("/answer", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

        
	return render_template("index.html", prediction = p, img_path = img_path, x=True)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = False)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
 
    
