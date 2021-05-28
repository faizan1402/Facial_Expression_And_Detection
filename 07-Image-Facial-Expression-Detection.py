# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:44:55 2020

@author: offic
"""


"""
  Image Facial Expression Recognition 
   # Detect Facial Expressions From an Image
   

"""

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

# Loading the to detect the image
image_to_detect = cv2.imread('images/faizan_ahmad_unknown_3.jpg')

#Load model and load the weights
face_exp_model =model_from_json(open("Data_Set/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights("Data_Set/facial_expression_model_weights.h5")
#declare the emotions label
emotions_label=('angry','disgust','fear','sad','sur0prise','neutral','happy','smile')

    
#detect the all faces in the image
#arguments are image ,no of times to upsample model
all_face_locations =face_recognition.face_locations(image_to_detect,model='hog')
#print the number of face detected
print('There are {} no of facts in this image'.format(len((all_face_locations))))
    
#looping through the face locations
for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get four position values of current face 
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        #printing the locations of current face index
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        #slicing the current face from from main image
        current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
        #draw rectangle around the face detected
        cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),3)
        #Preprocess input ,convert it to an image like as the dataset
        #Convert to grayscale
        
        current_face_image = cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)
    
        #resize to 48X48 pixel size
        current_face_image =cv2.resize(current_face_image,(48,48))
    
        #convert the PIL(python image library) into 3-D numpy array
        img_pixels =image.img_to_array(current_face_image)
        #expand the shape of an array into single row multiple columns
        img_pixels =np.expand_dims(img_pixels,axis=0)
        #pixels are in range of [0,255].normalize all pixels in sclae of [0,1]
        img_pixels /=255
        #Do prediction and model,get the prediction values for all 7 epressions
        exp_predictions = face_exp_model.predict(img_pixels)
        #Find max indexed prediction value (0,till,7)
        max_index = np.argmax(exp_predictions[0])
        #get corresponding label  from emotions label
        emotion_label = emotions_label[max_index]
        
        #display the name as text in the image
        font =cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_to_detect,emotion_label,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
        
    #showing the current face with dynamic
    
cv2.imshow("Image Face Emotion",image_to_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()
