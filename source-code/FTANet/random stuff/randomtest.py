import numpy as np
from numpy import genfromtxt
import mir_eval
# from evaluator import melody_eval

# train_list_file = './database/retrain_list/train_file.txt'
# valid_list_file = './database/retrain_list/valid_file.txt'
def Convert(string):
    li = list(string.split(", "))
    return li

with open("./database/retrain_list/train_file_github.txt", 'r') as queryf0_list:
    train_list_file = queryf0_list.readlines()

with open("./database/retrain_list/valid_file_github.txt", 'r') as queryf0_list:
    valid_list_file = queryf0_list.readlines()

# print('train file')
# for f0 in train_list_file: 
#     f0 = f0.replace("\n","")
#     f0 = Convert(f0)
#     pitch=np.loadtxt(f0[1])
#     if len(pitch) != 250:
#         print(len(pitch))

#     wavpath = f0[1].replace('.pv', '.wav')
#     # print(wavpath)
# print('\nvalid file')
# for f0 in valid_list_file:
#     f0 = f0.replace("\n","")
#     f0 = Convert(f0)
#     pitch=np.loadtxt(f0[1])
#     if len(pitch) != 250:
#         print(len(pitch))

# evaluasi praat - mir-qbsh manual
avg_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
for f0 in valid_list_file:
    f0 = f0.replace("\n","")
    f0 = Convert(f0)
    praat = f0[0].replace('\\cfp_feature', '\\praat_result\\mir-qbsh')
    praat = praat.replace('.npy', '.csv')
    praat = praat.replace('_github', '')
    
    f0_manual = f0[0].replace('\\cfp_feature', '\\MIR-QBSH\\f0File')
    f0_manual = f0_manual.replace('.npy', '.csv')
    f0_manual = f0_manual.replace('_github', '')
    
    pitch_f0    = genfromtxt(f0_manual, delimiter=',')
    pitch_praat = genfromtxt(praat, delimiter=',')
    
    # print(np.where((pitch_praat[1:,0]) == 1.50)[0])
    # print(pitch_praat[149,1])
    pitch_praat_new = []
    time_praat_new = []
    
    for t in pitch_f0[1:]:
        time_praat_new.append(round(t[0], 2))
        index = np.where((pitch_praat[1:,0]) == round(t[0], 2))[0]
        if index.size == 0:
            pitch_praat_new.append(t[1])
        else:            
            pitch_praat_new.append(pitch_praat[int(index)+1,1])
        
    time_praat_new = np.reshape(time_praat_new, (len(time_praat_new), 1))
    pitch_praat_new = np.reshape(pitch_praat_new, (len(pitch_praat_new), 1))
    
    pitch_praat_valid = np.concatenate((time_praat_new, pitch_praat_new), axis=1)
    
    eval_arr = melody_eval(pitch_f0[1:], pitch_praat_valid)
    avg_eval_arr += eval_arr
avg_eval_arr /= len(valid_list_file)

print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))
