from flask import Flask, flash, redirect, url_for, render_template, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
import requests
from googlesearch import search
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, model_from_json
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import cv2
import json

app = Flask(__name__)

disease_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
				 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy',
				 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 
				 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 
				 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
				 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
				 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
				 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
				 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
				 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
				 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
				 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
				 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
				 ]



def search_plant_disease(disease):
	query = f"{disease} plant disease care tips"
	num_results = 5  
	search_results = search(query, num_results=num_results, lang='en')
	results = []
	for result in search_results:
		results.append(result)
	return results



def preprocess_data(input_data):
	return cv2.resize(input_data[0], dsize = (224, 224), interpolation=cv2.INTER_CUBIC) * (1.0 / 255)


@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		image_file = request.files['image']
		image_file.save("data.jpeg")
		
	img_arr = cv2.imread('data.jpeg')
	processed_img = preprocess_input(img_arr)
	input_data = np.expand_dims(processed_img, axis=0)
	f = open("model/model_arch.json", "r")
	model = model_from_json(f.read())
	model.load_weights("model/model_weights.h5")
	f.close()
	data = preprocess_data(input_data)
	
	predictions = model.predict(np.expand_dims(data, axis=0))
	predicted_index = np.argmax(predictions)
	predicted_label = disease_names[predicted_index]
	return jsonify(label=predicted_label)