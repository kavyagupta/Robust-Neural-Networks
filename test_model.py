from imports import *

def test_model(Net, x_test, file_save, S1, sr): 
    Net.summary()
    test_predict = Net.predict(x_test)

    #P_true = np.log(np.mean(np.square(x_test)))
    #P_predicted = np.log(np.mean(np.square(test_predict)))
    #print('power true',P_true)
    #print('power predicted',P_predicted)
    
    
    out1 = (S1/x_test.T)*test_predict.T
    test1_recons = librosa.istft(out1, win_length= 1024, hop_length=512)
    librosa.output.write_wav(file_save, test1_recons, sr)
    

