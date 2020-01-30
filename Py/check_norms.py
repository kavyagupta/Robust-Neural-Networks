import numpy as np
def check_norms(model):
    w = dict()
    for layer in model.layers:
        name = layer.get_config()['name']
        print(name)
        if name.startswith('dense'):
            w[name] = layer.get_weights()[0]   
    P = w['dense_1'] @ w['dense_2'] @ w['dense_3'] @ w['dense_4'] @ w['dense_5'] @ w['dense_6']
    norm = np.linalg.norm(P, ord=2)
    print('Lipschitz constant of the network is: ', norm)


