from imports import *
from keras import backend as K
from keras import *





'''Add a metric to compute a relative SNR -- can find something better '''
def SNR_compute(y_true, y_pred):
    inp = tf.convert_to_tensor(y_true, dtype=tf.float32)
    P_true = tf.math.log(tf.reduce_mean(tf.square(inp), 1))
    P_predicted = tf.math.log(tf.reduce_mean(tf.square(y_pred), 1))
    return (P_true - P_predicted)

'''Constraint for the norms '''
class Norm_Constraint (Callback):
    min_value = 0.0
    rho = 1
    nit = 10000
 #   layers = [ 1,5,9,13,17]
    layers = [0, 4, 8, 12, 16]
    
    def Constraint(self, w, A, B, cnst):
        gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
        Y = np.zeros([Net.layers[self.layers[-1]].output_shape[1], Net.layers[0].input_shape[1]])
        for _ in range(self.nit):
            w_new = w - (np.transpose(A) @ Y @ np.transpose(B))
           # w_new *= np.greater_equal(w_new, 0) # ensure non-negative weights
            for i in range(w_new.shape[0]):
                for j in range(w_new.shape[1]):
                    if (np.absolute(i - j) >= np.ceil(w_new.shape[0] / 2) ):
                        w_new[i][j] = 0
            T = A @ w_new @ B
            [u,s,v] = np.linalg.svd(T)
            criterion = np.linalg.norm(w_new - w, ord='fro')
            constraint = np.linalg.norm(s[s > self.rho] - self.rho, ord=2)
            #print( 'iteration:', i+1, 'criterion: ', criterion, 'constraint: ', constraint)
            Yt = Y + gam * T
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
    model_name = '../Models/test_new.h5'
    n_factor = 50
    file_save = "../Tests/piano_test3-{epoch:02d}.wav"
    
    Tw = 0.5
    Fs = 44100
        
    N = int(Tw * Fs)
    x_train, x_val, x_test, y_train, y_val, y_test, S1, sr = make_input(path, name, N, path_save )
    #x_train = x_train[0:2000, :]
    #y_train = y_train[0:2000, :]
    
    print('x_train', x_train.shape, 'x_val', x_val.shape, 'x_test', x_test.shape)
    print('y_train', y_train.shape, 'y_val', y_val.shape, 'y_test', y_test.shape)

    #x_train = x_train[:,0:500]
    #y_train = y_train[:,0:500]
    
    #filepath= "./checkpoints/piano2-{epoch:02d}-{loss:.4f}.hdf5"
    
    # Scales the training and test data to range between 0 and 1.
    '''
    Define the network -- I think this should be later included in a different function, when we decide upon the architecture
    '''
    
    input_dim = x_train.shape[1] #513 #8033
    print(input_dim)
    
    Net = Sequential()
    
    Net.add(Dense(4*n_factor, input_shape=(input_dim,), activation='relu',kernel_initializer=my_init1))
    
    Net.add(Activation('relu'))
    Net.add(BatchNormalization())
    Net.add(Dropout(0.5))
    
    Net.add(Dense(2 * n_factor,kernel_initializer=my_init2))
    Net.add(Activation('relu'))
    Net.add(BatchNormalization())
    Net.add(Dropout(0.5))
    
    Net.add(Dense(n_factor,kernel_initializer=my_init3))
    Net.add(Activation('relu'))
    Net.add(BatchNormalization())
    Net.add(Dropout(0.5))
    
    Net.add(Dense(2*n_factor,kernel_initializer=my_init4))
    Net.add(Activation('relu'))
    Net.add(BatchNormalization())
    Net.add(Dropout(0.5))
    
    Net.add(Dense(4*n_factor,kernel_initializer=my_init5))
    Net.add(Activation('relu'))
    Net.add(BatchNormalization())
    Net.add(Dropout(0.5))
    
    Net.add(Dense(input_dim,kernel_initializer=my_init6))
    Net.add(Activation('sigmoid'))
    Net.add(BatchNormalization())
    
    Net.summary()
    
    
    checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
    cnstr_weight = Norm_Constraint()
    
    Net.compile(optimizer='adam', loss='mse', metrics=['mae', SNR_compute])
    Net.fit(x_train, y_train, epochs=10, batch_size=512, shuffle=True, validation_data=(x_val, y_val), callbacks=[checkpoint, cnstr_weight])
    #Net = load_model(model_name, custom_objects={'Norm_Constraint': Norm_Constraint, 'SNR_compute': SNR_compute})
    #Check if the constraint holds - also computes L constant - it's kinda hardcoded, must be changes if the net architecture changes
    #Net.load_weights('weights.h5')
    check_norms(Net)

    #Save the weights for further checks
    save_weights(Net)
    
    #Test the model -- this should be modified to compute a SNR or some kind of performance metric as well
    test_model(Net, x_test, file_save, S1, sr)
    
    
