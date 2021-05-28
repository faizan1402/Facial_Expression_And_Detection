# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:11:36 2020

@author: offic
"""


"""
             #Real Time Facial Expression Recognition
# Detect Facial Expressions froam Real time Webcam Video
              
              # Installing Tensor Flow and Keras in Anaconda
  -> Using the Nural network Library Backend tensor flow
         https://www.tensorflow.org/
  -> Keras ,an eassy to use wrapper to the low-level library Tensor flow
        https://www.keras.io/
  -> Before installing Keras ,install the backend engine : TensorFlow
  -> Close all open spyder or navigator window and open Anaconda prompt
      conda install tensor flow
 -> Sourece of facial Expression  Data
  . The Kaggle Facial Expression recognition challenge Dataset
        https://www.kaggle.com/c./challenges-in-representation-learning-facial-expression-recognition-challenge
  . Consits of 48*48 pixel grayscscale face images
  . Each images corresponding 1 of 7 expression categories
  .whole dataset contains approximately 36,000 images
  
    Input layer  means base network to higher network
    Input layer1
    Hidden layer2 
    Output
  Dataset in the folder 'Dataset'
   
  Model Structure file : facial_expression_model_structure.json
  Model Weight file:  facial_expression_model_weight.h5
  
 
         # Real time Facial Expression Detection
   Step 1: Import  the libraies
    #importing the requied libraies
   import numpy as np
   import cv2
   from keras.preprocessing import image
   from keras.models import model_from_json
   
# Capture the video from default camera
   webcam_video_stream =cv2.videoCapture(0)

Srep 2: Initialize model and load weights
  
# capture the video from default camera
webcam_video_stream =cv2.VideoCapture(0)

#face expression model initilization
   face_exp_model =model_from_json(open('data-set/facial_expression_model_structure.json','r').read())
   
   #load weights into model 
   
   face_exp_model.load_weights('data-set/facial_expression_model_weights.h5')
   
   #list of emotions labels
    emotions_label =('angry','fear','happy','surprise','neutral')

   # Real-time Facial Expression Dtection
  Step 3 :
        Slice and extract the face -Image from frame(remove the bluring script)
        
        #slicing the current face from main image
    
       currnt_face_image =current_frame[top_pos:bottom_pos,left_pos,right_pos]
       
       #blur the blurred face into the actual frame
       
       current_frame[top_pos:bottom_pos,left_pos:right_pos]= current_face_image
       
   Step 4 pre-process  the input image(video frame)
         
     For Example,to normalize the range 0 to 10
       Divide all values by 10 and will get range as 0 to 1
       Exampl : 0,1,2,3,....10 will be 0.0,0.1,0.2,.....10
     
       #draw recatangle around the face
     
     current_face_image =current_frame[top_pos:bottom_pos.left_pos:right_pos]
     
     #preprocess input ,convert if it to an image like as the data dataset
      
     #convert to grayscale 
     current_face_image =cv2.cvtColor(current_face_image,cv2.COLORBGR2GRAY)
     
     #resize to 48X48 pixel size
      
       current_face_image =cv2.resize(current_face_image,(48,48))
       
      #convert the PIL(python image library) into 3-D numpy array
       
       img_pixels =image.img_to_array(current_face_image)
       #expand the shape of an array into single row multiple columns
       img_pixels =np.expand_dims(img_pixels,axis=0)
       
       #pixels are in range of [0,255].normalize all pixels in sclae of [0,1]
       
        img_pixels /=255
    
     Step 5 Do prediction and model,get the prediction values for all 7 epressions
        exp_predictions = face_exp_model.predict(img_pixels)
        
        # Find max indexed prediction value (0,till,7)
        
        max_index =np.argmax(exp_predictions[0])
        # get corresponding lable from emotions_label
        
        emotion_label =emotions_label[max_index]
        
        # display the name as text in the image
        front =cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,emotion_label,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
        # display the video
        cv2.imshow("Webcam ",current_frame)
        

              
"""

import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition


# default camera indexe
webcam_video_stream = cv2.VideoCapture(0)
#Load model and load the weights
face_exp_model = model_from_json(open("data_set/facial_expression_model_structure.json","r",encoding="utf-8").read())
face_exp_model.load_weights("data_set/facial_expression_model_weights.h5")
#declare the emotions label
emotions_label=('angry','disgust','fear','sad','surprise','neutral')


#initialize the array variable to hold all face locations in the frame

all_face_locations =[]     

#loop through ecery frame in the video
while True:#means the no face detector true means 
    #get the current frame from the video stream as an image
    ret,current_frame =webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small =cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    
    #detect the all faces in the image
    #arguments are image ,no of times to upsample model
    
    all_face_locations =face_recognition.face_locations(current_frame_small,model='hog')
    
    #looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get four position values of current face 
        
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #change the position magnitude to fit the actual size video frame
     
        top_pos = top_pos*4
        right_pos =right_pos*4
        bottom_pos =bottom_pos*4
        left_pos =left_pos*4
    
        #printing the locations of current face index
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        #Extract the face from the from,blur it paste it back to the frame
        #slicing the current face from from main image
        current_face_image =current_frame[top_pos:bottom_pos,left_pos:right_pos]
    
        #draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),3)
    
    
        #Preprocess input ,convert it to an image like as the dataset
        #Convert to grayscale
        current_face_image =cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GAY)
    
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
        
        #get corresponding label  from emotions label
        exp_predictions=face
        emotion_label =emotions_label[max_index]
        
        #display the name as text in the image
        front =cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame,emotion_label,(left_pos,bottom_pos),font,0.5,(255,255,255),1)
        
    #showing the current face with dynamic
    
    cv2.imshow("Web Cam video",current_frame)
    
   
# this is 0xFF is a 32 bit 
    if cv2.waitKey(1) & 0xFF ==ord('q'):# so is very imp other wise my camera always on not closed so very imp to close the camera
        
        break

webcam_video_stream.release()# this function to release the camera 
cv2.destroyAllWindows()
