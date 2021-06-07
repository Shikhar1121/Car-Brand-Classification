# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:03:34 2021

@author: Shikhar
"""
from tensorflow.keras.layers import Input,Lambda,Dense,Flatten
from tensorflow.keras.models import Model

#resnet50 is a pretrained cnn model
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

from tensorflow.keras.models import  Sequential

import numpy as np

import matplotlib.pyplot as plt

from glob import glob



#resizing image sizes to 224,224 as resnet50 takes input images sized 224,224
image_size = [224,224]

#Trainig Path
train_path = 'Data/Train'

#test path
test_path = 'Data/Test'


#initializing resnet50
# +[3] because the images provided are in 3 channels rgb
#include top = false because we dont want the already trained input w will provide our input
resnet = ResNet50(input_shape = image_size+[3], weights = 'imagenet',include_top = False)


# the weights in resnet are used so no need to train and reinitialize the weights just train 
#the last layer 

for layers in resnet.layers:
    layers.trainable = False

# tells us how many folders are present
folders = glob('Data/Train/*')

#flattening the outplut from resnet layers
x = Flatten()(resnet.output)


#using dense model to set the length of the folders as output
prediction =Dense(len(folders),activation='softmax')(x)

#creating model object
model = Model(resnet.input,outputs = prediction)

model.summary()


#compiling the model and telling it what cost and optiization it is going to use
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

#reading images from the data folder
#doing data augmentation it means it tries to create more images from the provided images
#as variation
# rescale means all the image pixels  will be rescaled by dividing each rgb channels by 255
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

#in test data never use data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)


#class mode is used because there are more than 2 classes if 2 classes use binary
trainig_set = train_datagen.flow_from_directory('Data/Train',target_size = (224,224) ,batch_size = 32,class_mode = 'categorical')

#for test dataset
test_set = test_datagen.flow_from_directory('Data/Test',target_size = (224,224) ,batch_size = 32,class_mode = 'categorical')

#fitting 
r = model.fit_generator(trainig_set,validation_data=test_set,epochs=50,steps_per_epoch=len(trainig_set),validation_steps=len(test_set))


#prediction
y_pred = model.predict(test_set)

#taking value from y_pred which are max
y_pred = np.argmax(y_pred,axis = 1)

#saving model
model.save('model_resnet50.h5')


