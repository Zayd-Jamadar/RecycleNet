TRAIN_DIR = './dataset/train'
# CHECKPOINT_PATH = './checkpoint/training_1'
# LATEST_CHECKPOINT = './checkpoint/15/11/2020--18:55/'

EPOCHS = 10
STEPS_PER_EPOCH = 30
BATCH_SIZE = 32

import numpy as np
import os
import time
from resnet50 import ResNet50

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb_classes = 5
image_input = Input(shape=(224, 224, 3))

partial_path = "./trained_models/partial/"
# partial_dir = os.path.dirname(partial_path)

def train_model():
    model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
    last_layer = model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(5, name='output_layer')(x)

    custom_resnet_model = Model(inputs=image_input, outputs=out)

    # custom_resnet_model = get_model()
    custom_resnet_model.summary()

    # Comment these two lines before training
    # for layer in model.layers[:-1]:
    #     layer.trainable=False

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator=train_datagen.flow_from_directory(
                                TRAIN_DIR,
                                target_size=(224, 224),
                                batch_size=BATCH_SIZE,
                                class_mode='categorical')



    # cp_callback=keras.callbacks.ModelCheckpoint(
    #     filepath=CHECKPOINT_PATH,
    #     verbose=1,
    #     save_weights_only=True,
    #     save_best_only=True,
    #     monitor='accuracy'
    # )

    opt = tfa.optimizers.SGDW(learning_rate = 0.01, 
                            weight_decay=0.0001,
                            momentum=0.9)

    custom_resnet_model.compile(loss='categorical_crossentropy',
                                optimizer=opt,
                                metrics=['accuracy'],)

    history = custom_resnet_model.fit(                    
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        # callbacks=[cp_callback]
    )

    # custom_resnet_model.save_weights(partial_dir)
    custom_resnet_model.save(partial_path)

def get_model():
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # model = ResNet50(input_tensor=image_input, include_top=True, weights=None)
    model = tf.keras.models.load_model(partial_path)
    last_layer = model.get_layer('activation_48').output
    custom_resnet_model = Model(inputs=image_input, outputs=last_layer)
    custom_resnet_model.summary()
    return custom_resnet_model

if __name__ == '__main__':
    train_model()
    # get_model()

