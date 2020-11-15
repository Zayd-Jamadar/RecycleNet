DATASET_DIR = './dataset'
CHECKPOINT_PATH = './checkpoint/'
EPOCHS = 10
STEPS_PER_EPOCH = 30
BATCH_SIZE = 32

import numpy as np
import os
import time
from resnet50 import ResNet50

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb_classes = 5
image_input = Input(shape=(224, 224, 3))

def train_model():
    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')

    model.summary()

    # Comment these two lines before training
    # for layer in model.layers[:-1]:
    #     layer.trainable=False

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(
                                DATASET_DIR,
                                target_size=(224, 224),
                                batch_size=BATCH_SIZE,
                                class_mode='categorical')

    cp_callback=keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        monitor='accuracy'
    )

    model.compile(loss='hinge',
                                optimizer='adadelta',
                                metrics=['accuracy'], )

    history = model.fit(                    
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[cp_callback]
    )

def get_model():
    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    last_layer = model.get_layer('avg_pool').output
    custom_resnet_model = Model(inputs=image_input, outputs=last_layer)
    return custom_resnet_model

if __name__ == '__main__':
    train_model()

