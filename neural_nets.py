# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:44:17 2020

@author: Dani
"""
import tensorflow as tf
import pdb
import numpy as np

"""
This function will initialize a unet model
input: 
    img: input image 
    @type: 3D ndarray
    num_classes: number of classes in input
    @type: int
    num_level: number of levels in UNet
    @type: int
    num_layers: number of convolutional layers at each level
    @type: int
    filter_size_start: ???
    kernal_size: size of kernal
    @type: list(int)
    
"""

class UNet:
    def __init__(self, img):
        self.img = img
        self.model = self.initalize_unet(self.img)
        

    def initalize_unet(self, img):
    
        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = img.shape
        img = np.moveaxis(img,0,2)
       
        
        #Build the model

        inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
        #Convert image integer values into floating point values
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction Path
        c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2,2))(c1) 
        
        c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)  
        
        c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)  
        
        c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)  
        
        c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        #Expansive path
        u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
        """
        TODO: dimensions size of c4 is 1 pixel moe than u6 axis = 1
        """
        paddings = tf.constant([[0, 0], [1, 0], [0,0],[0,0]])
        

        u6 = tf.pad(u6, paddings, mode="CONSTANT")
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        
        paddings = tf.constant([[0, 0], [0, 0], [1,0],[0,0]])

        u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
        u7 = tf.pad(u7, paddings, mode="CONSTANT")

        u7 = tf.keras.layers.concatenate([u7, c3])

        c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)

        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)  
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)
        
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs] )
     
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
       # model.summary()
    
    

    
    
    
 
    




    
    
    
    