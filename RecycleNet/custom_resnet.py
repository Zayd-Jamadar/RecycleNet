TRAIN_DIR = './dataset/train'
# CHECKPOINT_PATH = './checkpoint/training_1'
# LATEST_CHECKPOINT = './checkpoint/15/11/2020--18:55/'
LOGS_DIR = './logs/'

EPOCHS = 12
STEPS_PER_EPOCH = 30
BATCH_SIZE = 12

import numpy as np
import os
import datetime
import time
from resnet50 import ResNet50
import matplotlib.pyplot as plt

import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb_classes = 5
image_input = Input(shape=(224, 224, 3))

partial_path = "./trained_models/partial/"
# partial_dir = os.path.dirname(partial_path)

def plot_graph(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

def train_model():
    model = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')
    last_layer = model.get_layer('avg_pool').output
    x = Flatten()(last_layer)
    out = Dense(5, name='output_layer', activation=tf.nn.softmax)(x)

    custom_resnet_model = Model(inputs=image_input, outputs=out)

    # custom_resnet_model = get_model()
    custom_resnet_model.summary()

    #Comment these two lines before training
    for layer in custom_resnet_model.layers[:-1]:
        layer.trainable=False

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    train_generator=train_datagen.flow_from_directory(
                                TRAIN_DIR,
                                target_size=(224, 224),
                                batch_size=BATCH_SIZE,
                                class_mode='sparse')



    # cp_callback=keras.callbacks.ModelCheckpoint(
    #     filepath=CHECKPOINT_PATH,
    #     verbose=1,
    #     save_weights_only=True,
    #     save_best_only=True,
    #     monitor='accuracy'
    # )

    # opt = tfa.optimizers.SGDW(learning_rate = 0.01, 
    #                         weight_decay=0.0001,
    #                         momentum=0.9)

    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')

    log_dir = LOGS_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    custom_resnet_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt,
                                metrics=['accuracy'],)

    history = custom_resnet_model.fit(                    
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback]
    )
    plot_graph(history)

    # custom_resnet_model.save_weights(partial_dir)
    custom_resnet_model.save(partial_path)

def get_model():
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # model = ResNet50(input_tensor=image_input, include_top=True, weights=None)
    # model = keras.models.load_model(partial_path)
    # m = keras.Sequential()
    # for l in model.layers[0:176]:
    #     m.add(l)

    #     last_layer = model.get_layer('activation_48').output
    # custom_resnet_model = Model(inputs=image_input, outputs=last_layer)
    # custom_resnet_model.summary()

    model = keras.models.load_model(partial_path)
    layer_name = 'res5c_branch2c'
    custom_resnet_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    custom_resnet_model.summary()
    return custom_resnet_model

if __name__ == '__main__':
    train_model()
    # get_model()

