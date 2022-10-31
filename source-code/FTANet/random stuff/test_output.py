# from loader import get_CenFreq, seq2map
import sys
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')
from numpy import genfromtxt
import numpy as np

# train_list_file = 'E:\Kuliah\S2\Thesis\source code thesis\database\\retraining\\train_file_2007.txt'
# with open(train_list_file) as f:
#     feature_files = f.readlines()

# fname = feature_files[0].replace("\n","")
# fname = fname.split(", ")

# pitch = genfromtxt(fname[1], delimiter=',')
# pitch = pitch[1:,1] # for praat 
            
            
# CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60) # (321) #参数是特征提取时就固定的
# print(pitch)
# for index in range(len(pitch)):
#     if pitch[index] != 0:
#         pos = (np.abs(CenFreq-pitch[index])).argmin()        
#         pitch[index] = CenFreq[pos]
    

# mapping = seq2map(pitch, CenFreq)
# # # CenFreq = np.array(CenFreq)
# print(len(pitch))
# print(len(CenFreq))
# print(len(mapping[13]))
# print(np.argmax(mapping[13]))
# print(np.where(CenFreq == pitch[13]))


# pitch_range = np.arange(38, 83 + 1.0/16, 1.0/16)
# pitch_range = np.concatenate([np.zeros(1), pitch_range])


# note = []
# for pitch in pitch_range:
#     if pitch> 0:
#         note.append(round(2 ** ((pitch - 69) / 12.) * 440, 3))
#     else:
#         note.append(pitch)

# print(note)
# print(pitch_range)
# print(pitch_range.shape)
# for x in mapping:
#     print(x)

# with open(".\database\MIR-QBSH\querywav_list.txt",'r') as f:
#     querylist = f.readlines()
