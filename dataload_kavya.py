import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import resample
from scipy import signal
import random
import librosa

def make_input(path, name, N, path_save):
    folders = os.listdir(path)
    #x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []
    X_train = np.empty((0, 513))
    Y_train = np.empty((0, 513))

    X_val = np.empty((0, 513))
    Y_val = np.empty((0, 513))

    X_test = np.empty((0, 513))
    Y_test = np.empty((0, 513))

    for folder in folders:
        data = os.path.join(path, folder)
        files = os.listdir(data)
        
        #files = files[0:20]

        for file in files:
            if file.split('.')[-1] == 'wav':
                f = os.path.join(data, file)
                Fs, audio = wavfile.read(f)

                
                audio = (audio[..., 0] + audio[...,1]) / 2 # averaged two channels -- maybe we cn only take one 
                audio = audio / np.max(np.abs(audio))  # norm the signal bwt 0 and 1
                if folder == 'Test':
                    wavfile.write('../Tests/test_clean.wav', Fs, audio)
                    print(audio.shape)
                pth = data + '/' + str(file.split('.')[0]) + '.txt'
                
                with open(pth, 'rt') as x:
                    lines = x.readlines()
                '''
                Add noise to the data choosing a SNR for the training signals bwt 5 and 30 dB
                Noise is AWGN
                '''
                target_snr_db = random.randrange(5, 30, 1)
                sample = np.asanyarray(audio)

                signal_avg_watts = np.mean(sample ** 2)
               # print('singal_watts ', signal_avg_watts)
                signal_avg_db = 10 * np.log10(signal_avg_watts )
               # print('signal_db ' , signal_avg_db)
                noise_avg_db = signal_avg_db - target_snr_db
               # print('noise_avg_db ', noise_avg_db)
                noise_avg_watts = 10 ** (noise_avg_db / 10)
          #      print('noise_avg_watts ', noise_avg_watts)
                mean_noise = 0

                k = np.sqrt(1 / (10 ** (target_snr_db / 10)))
                noise_volts = k * np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sample))
                noisy_signal = sample + noise_volts

                pure_audio = audio
                audio = noisy_signal

                S=librosa.stft(pure_audio, n_fft=1024, hop_length=512)
                X=librosa.stft(audio, n_fft=1024, hop_length=512)

                abs_S = np.abs(S.T)
                #print(abs_S.shape)
                abs_X = np.abs(X.T)

                if folder == 'Train':

                  X_train = np.append(X_train, abs_X, axis=0)
                  Y_train = np.append(Y_train, abs_S, axis=0)

                elif folder == 'Val':
                  X_val = np.append(X_val, abs_X, axis=0)
                  Y_val= np.append(Y_val, abs_S, axis=0)


                elif folder == 'Test':
                  X_test = np.append(X_test, abs_X, axis=0)
                  Y_test = np.append(Y_test, abs_S, axis=0)


                if folder == 'Test':
                  S1 = X
                  F_S1 = Fs
                  #S1_1 = librosa.istft(S1, win_length= 1024, hop_length=512)
                  librosa.output.write_wav('../Tests/test_load.wav', audio, Fs)
                
                '''
                for i in range(len(x_train)):
                  a = x_train[i]
                  b = y_train[i]
                  X_train = np.append(X_train, a, axis=0)
                  Y_train = np.append(Y_train, b, axis=0)


                
                for i in range(len(x_val)):
                  a = x_val[i]
                  b = y_val[i]
                  X_val = np.append(X_val, a, axis=0)
                  Y_val= np.append(Y_val, b, axis=0)


                
                for i in range(len(x_test)):
                  a = x_test[i]
                  b = y_test[i]
                  X_test = np.append(X_test, a, axis=0)
                  Y_test = np.append(Y_test, b, axis=0)
          #     print('noisy signal snr = ', target_snr_db, k)
             #   path1 ='tests/test_noisy' + file  
                '''
                '''
                if folder == 'Test':
                    wavfile.write('../Tests/test_noisy.wav', Fs, audio) 

                for i in range (0, audio.size-N, N):
                    fragment = audio[i: (i + N)]
                    pure_fragment = pure_audio[i: (i + N)]

                    if folder == 'Train':
                         x_train.append(fragment)
                         y_train.append(pure_fragment)

                    elif folder == 'Val':
                         x_val.append(fragment)
                         y_val.append(pure_fragment)

                    elif folder == 'Test':
                         x_test.append(fragment)
                         y_test.append(pure_fragment)
                  '''

    #x_train, x_val, x_test, y_train, y_val, y_test = np.asanyarray(x_train), np.asanyarray(x_val), np.asanyarray(x_test), np.asanyarray(y_train), np.asanyarray(y_val), np.asanyarray(y_test)
    #print(x_train.dtype)
    #x_train, x_val, x_test, y_train, y_val, y_test = x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32), y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)
    print('x_train', X_train.shape, 'x_val', X_val.shape, 'x_test', X_test.shape)
    print('y_train', Y_train.shape, 'y_val', Y_val.shape, 'y_test', Y_test.shape)
    
    np.save((path_save + 'x_train_{}'.format(name)), X_train)
    np.save((path_save + 'y_train_{}'.format(name)), Y_train)
    np.save((path_save + 'x_val_{}'.format(name)), X_val)
    np.save((path_save + 'y_val_{}'.format(name)), Y_val)
    np.save((path_save + 'x_test_{}'.format(name)), X_test)
    np.save((path_save + 'y_test_{}'.format(name)), Y_test)
    np.save((path_save + 'S1_{}'.format(name)), S1)
    np.save((path_save + 'sr_{}'.format(name)), F_S1)
    
    #y =  np.load(path_save + 'x_test_{}.npy'.format(name))
   # print(y.dtype)
   # print(y.shape)
   # print(y.size)
   # print(y[0].shape)
   # print(np.allclose(x_test,y))
    #wavfile.write('../Tests/test_load.wav', Fs, np.reshape(y, y.size))
    
    return X_train, X_val, X_test, X_train, Y_val, Y_test , S1, F_S1

'''
if __name__=='__main__':
    path ='../Music_Database'
    path_save = '../Save/'
    name = 'data_0.5window'

    Tw = 0.5
    Fs = 44100
    
    N = int(Tw * Fs)
    print(N)

    make_input(path, name, N, path_save )
'''
