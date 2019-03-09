# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:32:36 2019

@author: Praveen

"""
# Convolution Neural Networks
# Part1 - Building CNN

#importing keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# step 1 : Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation= 'relu'))

#Here since we are using Tflow as backend, the input_shape should be given as (rows, columns, channels)
#but if using theano as backend then input_shape=(channels, rows, columns) for each image
# channels =3 if colour image and channels = 2 if Black and white image

# step 2 : pooling. Here we are pooling the feature maps and divide them by 2, thereby reducing
#the number of feature maps and also retaining the info in the images and ensuring the perf. of
#the model.

classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3: Flattening
#Here we are converting the pooled feature maps into a giant 1-D vector to use as input nodes
# for our ANN.
classifier.add(Flatten())

#Step 4: Full connection
classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dense(units = 1,activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy',metrics=['accuracy'])

#Part 2: Fitting the CNN to the images
#uses preprocessing of the images, called image augmentation using keras library
#if we dont use image augmentation, the model is overfitted ie., accuracy will be high
#on training dataset but poor on test dataset
#Data augmentation can be applied not only to images but to sequences and text data also. This
#helps when there is less data for training. For Images, the model needs very high no. of images
#to find the correlations, learn patterns and features, this data augmentation can help if we 
#have less training images like here we have only 8000 Dogs and cats training images.

#using the code from keras documentation website:
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(training_set,
                            steps_per_epoch=8000,
                            epochs=5,
                            validation_data=test_set,
                            validation_steps=2000)

#To increase the accuracy on test set, we need to add more convolution layers. Adding 1 additonal
#conv layer will increase the test accuracy significantly but you can add another fully connected
#layer also in addition to conv layer

# We can increase the target size param also from (64,64) to (128, 128) to increase the 
#test accuracy

#part 3: Prediction on single image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
#here using this test_image for prediction gives an error as the predict class expects
#the batch number in the input arguments along with the shape of the image, so we need to 
#expand the dimensions using 'expand_dims' function.
test_image =np.expand_dims(test_image, axis =0) #This will add a new dimension at first argument

classifier.predict_classes(test_image)




