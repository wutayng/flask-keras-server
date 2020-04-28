from flask import Flask, jsonify, send_file, render_template, request, make_response
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, sys, io, shutil, tempfile
import cv2
from wsgiref.util import FileWrapper

from config import *
from utils.load import *
from utils.image import *

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/api/v1.0/predictVideo', methods=['POST'])
def predictVideo():

	# Input File
	file_in = tempfile.NamedTemporaryFile()
	file_in.write(request.files['file'].read())
	vid = cv2.VideoCapture(file_in.name)

	# Temporary Output for VideoWriter
	file_out = tempfile.NamedTemporaryFile(suffix = '.mp4')

	# Load Model
	model = load_model('model/' + os.listdir('model')[0], compile=False)
	# Compile Model with MEANINGLESS Loss and Optimizer
	# So predict method does not produce (AttributeError: 'Model' object has no attribute 'loss')
	model.compile(loss='mean_squared_error', optimizer='adam')

	initialize = True
	while(True): 
		# Reading from Frame 
		ret,frame = vid.read() 
		if ret: 
			# Continue Creating Images for Length of Video
			img = shape(frame, IMG_RESOLUTION)
			# Convert Image to Tensor
			pred_tensor = img[np.newaxis, :]
			# Predict Image using Loaded Keras Model
			predictions = model.predict(pred_tensor)
			# Convert Prediction Arrays to Image Heatmaps
			heatmap_tensor = heatmaps(predictions, CMAP, VMIN)

			# Paste Heatmaps on Original Image
			heatmap_image = draw_heatmaps(Image.fromarray(frame), heatmap_tensor[0, :])

			# Create/Append Output Video
			if initialize == True:
				height , width , channels =  frame.shape
				video = cv2.VideoWriter(file_out.name,cv2.VideoWriter_fourcc(*'avc1'),20.0,(int(width),int(height)))
				video.write(np.asarray(heatmap_image))
				initialize = False
			else:
				video.write(np.asarray(heatmap_image))

		else: 
			break

	# Release Input/Output Videos
	video.release()
	vid.release()

	return make_response(send_file(file_out.name, mimetype="video/mp4"))



if __name__ == '__main__':
	app.run()

