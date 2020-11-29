TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
LOGS_DIR = './logs/'
EPOCHS = 12
STEPS_PER_EPOCH = 30
BATCH_SIZE = 32

import os
import datetime
import time
from resnet50 import ResNet50
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb_classes = 5
image_input = Input(shape=(224, 224, 3))

timestr = time.strftime("%Y-%m-%d_%H:%M:%S")
GraphDir = './imgs/graphs/'

partial_path = "./trained_models/partial/"

def train_model():
    model = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')
    last_layer = model.get_layer('avg_pool').output
    x = Flatten()(last_layer)
    out = Dense(5, name='output_layer', activation='softmax')(x)

    custom_resnet_model = Model(inputs=image_input, outputs=out)

    custom_resnet_model.summary()

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(
                                TRAIN_DIR,
                                target_size=(224, 224),
                                batch_size=BATCH_SIZE,
                                class_mode='sparse')

    validation_generator=test_datagen.flow_from_directory(
                                TEST_DIR,
                                target_size=(224, 224),
                                batch_size=BATCH_SIZE,
                                class_mode='sparse')

    # opt = tfa.optimizers.SGDW(learning_rate = 0.01,
    #                         weight_decay=0.0001,
    #                         momentum=0.9)

    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')

    log_dir = LOGS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    custom_resnet_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt,
                                metrics=['accuracy'],)

    history = custom_resnet_model.fit(                    
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[tensorboard_callback]
    )

    custom_resnet_model.save(partial_path)

def get_model():
    model = keras.models.load_model(partial_path)
    layer_name = 'res5c_branch2c'
    custom_resnet_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    custom_resnet_model.summary()
    return custom_resnet_model

if __name__ == '__main__':
    train_model()

