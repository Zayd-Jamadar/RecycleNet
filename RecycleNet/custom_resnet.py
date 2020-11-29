import os
import datetime
import time
from resnet50 import ResNet50
import matplotlib.pyplot as plt
import config

import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Activation, Flatten
from keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


nb_classes = 5
image_input = Input(shape=(224, 224, 3))

timestr = time.strftime("%Y-%m-%d_%H:%M:%S")

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
                                config.TRAIN_DIR,
                                target_size=(224, 224),
                                batch_size=config.BATCH_SIZE,
                                class_mode='sparse')

    validation_generator=test_datagen.flow_from_directory(
                                config.TEST_DIR,
                                target_size=(224, 224),
                                batch_size=config.BATCH_SIZE,
                                class_mode='sparse')

    # opt = tfa.optimizers.SGDW(learning_rate = 0.01,
    #                         weight_decay=0.0001,
    #                         momentum=0.9)

    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')

    log_dir = config.LOGS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    custom_resnet_model.compile(loss='sparse_categorical_crossentropy',
                                optimizer=opt,
                                metrics=['accuracy'],)

    custom_resnet_model.fit(                    
        train_generator,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        callbacks=[tensorboard_callback]
    )

    custom_resnet_model.save(config.TRAINED_MODEL_DIR_RESNET)

def get_model():
    model = keras.models.load_model(config.TRAINED_MODEL_DIR_RESNET)
    layer_name = 'res5c_branch2c'
    custom_resnet_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    custom_resnet_model.summary()
    return custom_resnet_model

if __name__ == '__main__':
    train_model()

