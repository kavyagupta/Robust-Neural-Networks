import numpy as np

def get_weights(model):
    w = dict()
    for layer in model.layers:
        name = layer.get_config()['name']
        if name.startswith('dense'):
            w[name] = layer.get_weights()[0]
    return w
