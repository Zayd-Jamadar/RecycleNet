CHECKPOINT_PATH = './checkpoint/'

import tensorflow as tf
from tensorflow import keras

def test():
    model = keras.models.load_model(CHECKPOINT_PATH)
    
