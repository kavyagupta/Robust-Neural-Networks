from imports import *

def my_init1(shape, dtype=None):
    w = K.zeros((513, 200))
    w1 = w['dense_1']
    W_l = K.concatenate((w1, w), 0)
    W_r = K.concatenate((w, w), 0)
    return K.concatenate((W_l, W_r),1)

def my_init2(shape, dtype=None):
    w = K.zeros((200, 100))
    w1 = w['dense_2']
    W_l = K.concatenate((w1, w), 0)
    W_r = K.concatenate((w, w), 0)
    return K.concatenate((W_l, W_r),1)

def my_init3(shape, dtype=None):
    w = K.zeros((100,50))
    w1 = w['dense_3']
    W_l = K.concatenate((w1, w), 0)
    W_r = K.concatenate((w, w), 0)
    return K.concatenate((W_l, W_r),1)

def my_init4(shape, dtype=None):
    w = K.zeros((50, 100))
    w1 = w['dense_4']
    W_l = K.concatenate((w1, w), 0)
    W_r = K.concatenate((w, w), 0)
    return K.concatenate((W_l, W_r),1)

def my_init5(shape, dtype=None):
    w = K.zeros((100, 200))
    w1 = w['dense_5']
    W_l = K.concatenate((w1, w), 0)
    W_r = K.concatenate((w, w), 0)
    return K.concatenate((W_l, W_r),1)

def my_init6(shape, dtype=None):
    w = K.zeros((200, 513))
    w1 = w['dense_6']
    W_l = K.concatenate((w1, w), 0)
    W_r = K.concatenate((w, w), 0)
    return K.concatenate((W_l, W_r),1)

def input_init(x):
    return tf.linalg.matmul(x, tf.eye(513, 1026))

def get_output(x):
    return  tf.linalg.matmul(x, tf.eye(1026, 513))


'''Constraint for the norms '''
def SNR_compute(y_true, y_pred):
    inp = tf.convert_to_tensor(y_true, dtype=tf.float32)
    P_true = tf.math.log(tf.reduce_mean(tf.square(inp), 1))
    p_predicted = tf.math.log(tf.reduce_mean(tf.square(y_pred),1))
    return (P_true - P_predicted)

class Norm_Constraint (Callback):
    min_value = 0.0
    rho = 1
    nit = 100
    layers = [2, 5, 8, 11, 14, 17]
    
    def sparse_matrix(self, w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                 if (np.absolute(i - j) >= np.ceil((np.maximum(w.shape[0], w.shape[1]) / 2))):
                    w[i][j] = 0
        return w

    def get_projection(self,w):
        new_block_list = []
        new_horiz_block_list = []
        vertical_blocks = np.vsplit(w, 2)
        for vertical_block in vertical_blocks:
            blocks = np.hsplit(vertical_block, 2)
            for block in blocks:
                new_block = self.sparse_matrix(block)
               # new_block *= np.greater_equal(new_block,0)
                new_block_list.append(new_block)
        for i in range(2):
            lim = 2
            new_horiz_block_list.append(np.hstack(new_block_list[lim*i:lim + lim*i]))       
        return(np.vstack(new_horiz_block_list))

    def Constraint(self, w, A, B, cnst):
        gam = 1.99 / (np.square(np.dot(np.linalg.norm(A, ord=2), np.linalg.norm(B, ord=2)) + np.spacing(1)))
        Y = np.zeros([Net.layers[self.layers[-1]].output_shape[1], Net.layers[2].input_shape[1]])
        for _ in range(self.nit):
            w_new = w - (np.transpose(A) @ Y @ np.transpose(B))
            w_new *= np.greater_equal(w_new, 0) # ensure non-negative weights
            w_new = self.get_projection(w_new)
            T = A @ w_new @ B
            np.where(np.isfinite(T), T, 0)
            [u,s,v] = np.linalg.svd(T)
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
            B = np.eye(Net.layers[2].input_shape[1])
            for index_weight_B in self.layers:
                if (index_weight_B < index_weight):
                     B = np.transpose(Net.layers[index_weight_B].get_weights()[0]) @ B
            w = np.transpose(Net.layers[index_weight].get_weights()[0])
            wf = Net.layers[index_weight].get_weights()
            wf[0] = np.transpose(self.Constraint(w, A, B, 0.1))
            Net.layers[index_weight].set_weights(wf)
            A = A @ np.transpose(Net.layers[index_weight].get_weights()[0])

if __name__=='__main__' :

    path ='../Music_Database'
    path_save = '../Save/'
    name = 'data_0.5window'
    model_name = '../Models/mimo_2channels_relu.h5'
    save_model = '../Models/mimo_2channels_relu1.h5'
    n_factor = 50
    file_save = "../Tests/test.wav"
    
    Tw = 0.5
    Fs = 44100
        
    N = int(Tw * Fs)
#    x_train, x_val, x_test, y_train, y_val, y_test, S1, sr = make_input(path, name, N, path_save )

    x_train, x_val =  np.load(path_save + 'x_train_{}.npy'.format(name)), np.load(path_save + 'x_val_{}.npy'.format(name))
    y_train, y_val = np.load(path_save + 'y_train_{}.npy'.format(name)), np.load(path_save + 'y_val_{}.npy'.format(name))
    x_test, y_test = np.load(path_save + 'x_test_{}.npy'.format(name)), np.load(path_save + 'y_test_{}.npy'.format(name))
    S1 = np.load(path_save + 'S1_{}.npy'.format(name))

   # x_train = x_train[0: 40000, :]
   # y_train = y_train[0: 40000, :]

   # x_val = x_val[0: 10000, :]
   # y_val = y_val[0: 10000, :]

    '''
    Define the network -- I think this should be later included in a different function, when we decide upon the architecture
    '''
  #  prev_model = load_model(save_model, custom_objects={'Norm_Constraint': Norm_Constraint, 'SNR_compute': SNR_compute})

  #  w = get_weights(prev_model)
    cnstr_weight = Norm_Constraint()

    inpu = Input(shape=(x_train.shape[1], ))
    init = Lambda(input_init, output_shape=(1026,))(inpu)

    hdn_1 = Dense(400)(init)
    hdn_1 = Activation('relu')(hdn_1)
    hdn_1 = BatchNormalization()(hdn_1)

    hdn_2 = Dense(200)(hdn_1)
    hdn_2 = Activation('relu')(hdn_2)
    hdn_2 = BatchNormalization()(hdn_2)

    hdn_3 = Dense(100)(hdn_2)
    hdn_3 = Activation('relu')(hdn_3)
    hdn_3 = BatchNormalization()(hdn_3)

    hdn_4 = Dense(200)(hdn_3)
    hdn_4 = Activation('relu')(hdn_4)
    hdn_4 = BatchNormalization()(hdn_4)

    hdn_5 = Dense(400)(hdn_4)
    hdn_5 = Activation('relu')(hdn_5)
    hdn_5 = BatchNormalization()(hdn_5)

    hdn_6 = Dense(1026)(hdn_5)
    hdn_6 = Activation('relu')(hdn_6)
    hdn_6 = BatchNormalization()(hdn_6)

    output = Lambda(get_output, output_shape=(x_train.shape[1],))(hdn_6)

    checkpoint = ModelCheckpoint(save_model, monitor='loss', verbose=1, save_best_only=True)
    
    Net = Model(input=inpu, output=output)
    Net.compile(optimizer='adam', loss='mse', metrics=['mae'])
    Net.summary()
    Net = load_model(model_name, custom_objects={'Norm_Constraint': Norm_Constraint, 'SNR_compute': SNR_compute, 'input_init': input_init, 'get_output': get_output, 'tf': tf})
    Net.fit(x_train, y_train, epochs=20, batch_size=1024, shuffle=True, validation_data=(x_val, y_val), callbacks=[checkpoint, cnstr_weight])
    check_norms(Net)
    save_weights(Net)
    
    #Test the model -- this should be modified to compute a SNR or some kind of performance metric as well
    #test_model(Net, x_test, file_save, S1, Fs)
