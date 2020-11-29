partial_path = "./trained_models/partial/"

import tensorflow as tf
from tensorflow import keras

def test():
    model = keras.models.load_model(partial_path)
    flag = 1
    assert(flag==1)
    
