from imports import *
from keras import backend as K
import h5py
from sklearn.utils.extmath import randomized_svd
import timeit

'''Add a metric to compute a relative SNR -- can find something better '''
def SNR_compute(y_true, y_pred):
    inp = tf.convert_to_tensor(y_true, dtype=tf.float32)
    P_true = tf.math.log(tf.reduce_mean(tf.square(inp),0))/tf.math.log(tf.constant(10, dtype=inp.dtype))
    P_predicted = tf.math.log(tf.reduce_mean(tf.square(y_pred),0))/tf.math.log(tf.constant(10, dtype=y_pred.dtype))
    return (P_true - P_predicted)

'''Constraint for the norms '''
class Norm_Constraint (Callback):
    min_value = 0.0
    rho = 1
    nit = 1000
 #   layers = [ 1,5,9,13,17]
    layers = [1, 5, 9, 13, 17, 21]
    
    def Constraint(self, w, A, B, cnst):
        gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
        Y = np.zeros([Net.layers[self.layers[-1]].output_shape[1], Net.layers[0].input_shape[1]])
        for _ in range(self.nit):
            w_new = w - (np.transpose(A) @ Y @ np.transpose(B))
           # w_new *= np.greater_equal(w_new, 0) # ensure non-negative weights
            for i in range(w_new.shape[0]):
                for j in range(w_new.shape[1]):
                    if (np.absolute(i - j) >= np.ceil(w_new.shape[0] / 2) ):
                        w_new[i][j] = np.spacing(1)

            T = A @ w_new @ B
            np.where(np.isfinite(T), T, 0)            
            [u,s,v] = np.linalg.svd(T)
            #u, s, v = randomized_svd(T, n_components = 10)
            criterion = np.linalg.norm(w_new - w, ord='fro')
            constraint = np.linalg.norm(s[s > self.rho] - self.rho, ord=2)
            #print( 'iteration:', i+1, 'criterion: ', criterion, 'constraint: ', constraint)
            Yt = Y + gam * T
            np.where(np.isfinite(Yt), Yt, 0)
            [u1, s1, v1] = np.linalg.svd(Yt / gam, full_matrices=False)
            s1 = np.clip(s1, 0, self.rho)
            Y = Yt - gam * np.dot(u1 * s1, v1)
            if (criterion < 100 and constraint < cnst):
                return w_new

        return w_new

    def on_batch_end(self, batch, logs={}):

        A = np.eye(Net.layers[self.layers[-1]].output_shape[1])
        for index_weight in np.flip(self.layers):
            B = np.eye(Net.layers[0].input_shape[1])
            for index_weight_B in self.layers:
                if (index_weight_B < index_weight):
                    B = np.transpose(Net.layers[index_weight_B].get_weights()[0]) @ B
            w = np.transpose(Net.layers[index_weight].get_weights()[0])
            wf = Net.layers[index_weight].get_weights()
            wf[0] = np.transpose(self.Constraint(w, A, B, 0.1))
            Net.layers[index_weight].set_weights(wf)
            A = A @ np.transpose(Net.layers[index_weight].get_weights()[0])


if __name__=='__main__' :

    filename = 'weights.h5'

    f = h5py.File(filename,'r')

    def my_init1(shape, dtype=None):
        W = f['w1'][:]
        W = np.transpose(W)
        #W = K.random_normal(shape, dtype=dtype)
        print(W.shape)
        return W

    def my_init2(shape, dtype=None):
        W = f['w2'][:]
        W = np.transpose(W)
        #W = K.random_normal(shape, dtype=dtype)
        print(W.shape)
        return W

    def my_init3(shape, dtype=None):
        W = f['w3'][:]
        W = np.transpose(W)
        #W = K.random_normal(shape, dtype=dtype)
        print(W.shape)
        return W

    def my_init4(shape, dtype=None):
        W = f['w4'][:]
        W = np.transpose(W)
        #W = K.random_normal(shape, dtype=dtype)
        print(W.shape)
        return W

    def my_init5(shape, dtype=None):
        W = f['w5'][:]
        W = np.transpose(W)
        #W = K.random_normal(shape, dtype=dtype)
        print(W.shape)
        return W

    def my_init6(shape, dtype=None):
        W = f['w6'][:]
        W = np.transpose(W)
        #W = K.random_normal(shape, dtype=dtype)
        print(W.shape)
        return W
    
    path ='../Music_Database'
    path_save = '../Save/'
    name = 'data_0.5window'
    model_name = '../Models/test_cnstr_full.h5'
    n_factor = 50
    file_save = "../Tests/piano_cnstr_full-{epoch:02d}.wav"
    
    Tw = 0.5
    Fs = 44100
        
    N = int(Tw * Fs)

    #x_train, x_val, x_test, y_train, y_val, y_test, S1, sr = make_input(path, name, N, path_save )
    x_train = np.load(path_save + 'x_train_{}'.format(name) + '.npy')
    x_val = np.load(path_save + 'x_val_{}'.format(name)+ '.npy')
    x_test = np.load(path_save + 'x_test_{}'.format(name)+ '.npy')

    y_train = np.load(path_save + 'y_train_{}'.format(name)+ '.npy')
    y_val = np.load(path_save + 'y_val_{}'.format(name)+ '.npy')
    y_test = np.load(path_save + 'y_test_{}'.format(name)+ '.npy')

    S1 = np.load((path_save + 'S1_{}'.format(name))+ '.npy')
    sr = np.load((path_save + 'sr_{}'.format(name))+ '.npy')

    
 
    print('here')

    #x_train = x_train[0:4000, :]
    #y_train = y_train[0:4000, :]
    #x_val = x_val[0:2000,:]
    #y_val = y_val[0:2000,:]
    
    print('x_train', x_train.shape, 'x_val', x_val.shape, 'x_test', x_test.shape)
    print('y_train', y_train.shape, 'y_val', y_val.shape, 'y_test', y_test.shape)
    
    #filepath= "./checkpoints/piano2-{epoch:02d}-{loss:.4f}.hdf5"
    
    # Scales the training and test data to range between 0 and 1.
    '''
    Define the network -- I think this should be later included in a different function, when we decide upon the architecture
    '''
    cnstr_weight = Norm_Constraint()
    
    inpu = Input(shape=(x_train.shape[1], ))

    hdn_1 = Dense(200)(inpu)
    hdn_1 = Activation('relu')(hdn_1)
    hdn_1 = BatchNormalization()(hdn_1)
    hdn_1 = Dropout(0.5)(hdn_1)

    hdn_2 = Dense(100)(hdn_1)
    hdn_2 = Activation('relu')(hdn_2)
    hdn_2 = BatchNormalization()(hdn_2)
    hdn_2 = Dropout(0.5)(hdn_2)

    hdn_3 = Dense(50)(hdn_2)
    hdn_3 = Activation('relu')(hdn_3)
    hdn_3 = BatchNormalization()(hdn_3)
    hdn_3 = Dropout(0.5)(hdn_3)

    hdn_4 = Dense(100)(hdn_3)
    hdn_4 = Activation('relu')(hdn_4)
    hdn_4 = BatchNormalization()(hdn_4)
    hdn_4 = Dropout(0.5)(hdn_4)

    hdn_5 = Dense(200)(hdn_4)
    hdn_5 = Activation('relu')(hdn_5)
    hdn_5 = BatchNormalization()(hdn_5)
    hdn_5 = Dropout(0.5)(hdn_5)

    output = Dense(x_train.shape[1])(hdn_5)
    output = Activation('sigmoid')(output)
    output = BatchNormalization()(output)
    
    
    checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
    
    Net = Model(input=inpu, output=output)
    start = timeit.default_timer()
    #Net.load_weights('../Models/test_cnstr.h5')
    #Net = load_model('../Models/test_cnstr.h5', custom_objects={'Norm_Constraint': Norm_Constraint, 'SNR_compute': SNR_compute})
    Net.compile(optimizer='adam', loss='mse', metrics=['mae', SNR_compute])
    Net.fit(x_train, y_train, epochs=20, batch_size=512, shuffle=True, validation_data=(x_val, y_val), callbacks=[checkpoint,cnstr_weight])
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    #Net = load_model(model_name, custom_objects={'Norm_Constraint': Norm_Constraint, 'SNR_compute': SNR_compute})
    #Check if the constraint holds - also computes L constant - it's kinda hardcoded, must be changes if the net architecture changes
    
    check_norms(Net)

    #Save the weights for further checks
    save_weights(Net)
    test_model(Net,x_test,file_save,S1,sr)
    #Test the model -- this should be modified to compute a SNR or some kind of performance metric as well
    #test_model(Net, x_test, file_save, S1, sr)
    
    
