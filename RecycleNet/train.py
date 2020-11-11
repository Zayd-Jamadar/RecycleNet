DATASET_DIR = 'C:/Users/Zayd/RecycleNet/RecycleNet/dataset'

import numpy as np
import os
import time
from resnet50 import ResNet50

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb_classes = 5
image_input = Input(shape=(224, 224, 3))



def train_model():
    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    last_layer = model.get_layer('avg_pool').output
    x = Dense(5, name='output_layer',kernel_regularizer=l2(l=0.1))(last_layer)
    out = Activation('linear')(x)

    custom_resnet_model = Model(inputs=image_input, outputs=out)

    custom_resnet_model.summary()

    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable=False

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(
                                DATASET_DIR,
                                target_size=(224, 224),
                                batch_size=128,
                                class_mode='categorical')

    custom_resnet_model.compile(loss='hinge',
                                optimizer='adam',
                                metrics='accuracy', )

    custom_resnet_model.fit(                    
        train_generator,
        steps_per_epoch=2000,
        epochs=10
    )



if __name__ == '__main__':
    train_model()

