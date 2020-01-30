from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.models import Model
import numpy as np

def save_weights(model):
    for layer in model.layers:
        print(layer.get_config())
        name = layer.get_config()['name']
        if name.startswith('dense'):
            np.savetxt('weights/weights_%s' % name, layer.get_weights()[0], delimiter=' ')
            np.savetxt('weights/bias_%s' % name, layer.get_weights()[1], delimiter=' ')

