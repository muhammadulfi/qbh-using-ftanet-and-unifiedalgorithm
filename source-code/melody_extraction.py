import numpy as np
import pandas as pd
import parselmouth
import time
import os

# querywav_list = './database/MIR-QBSH/querywav.list'
totaltime = 0

# MIR-QBSH database
# with open(".\database\MIR-QBSH\querywav_list.txt",'r') as f:
#     querylist = f.readlines()
#     for row in querylist:        
#         start = time.time()
        
#         # prepare the output file
#         file = row.split("\\")
#         filename = '-'.join(file[-3:])
#         output_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\result_praat\mir-qbsh', filename + ".csv")
#         output_file = output_file.replace(".wav","")
#         output_file = output_file.replace("\n","")
        
#         # prepare query path 
#         query = '/'.join(file[-6:])
#         query = query.replace("\n","")
#         query = '/' + query
        
#         # start melody extraction
#         snd = parselmouth.Sound(query)
#         pitch = snd.to_pitch()
#         pitch_values = pitch.selected_array['frequency']

#         mel_time = []
#         f0 = []
        
#         for i in range(len(pitch_values)):
#             f0.append(pitch_values[i].round(decimals=3))
#             mel_time.append(pitch.xs()[i].round(decimals=3))
        
#         pd.DataFrame({'time':mel_time, 'f0':f0}).to_csv(output_file, index=False)
#         end = time.time()
#         totaltime += round(end-start, 3)
#     avg_time = totaltime/len(querylist)    
# f.close()

# ioacas database 
with open(".\database\IOACAS_QBH\querywav_list.txt",'r') as f:
    querylist = f.readlines()
    for row in querylist:        
        start = time.time()
        
        # prepare the output file
        file = row.split("\\")
        filename = file[-1]
        output_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\result_praat\ioacas', filename + ".csv")
        output_file = output_file.replace(".wav","")
        output_file = output_file.replace("\n","")
        
        # prepare query path 
        query = row.replace("\n","")
        
        # start melody extraction
        snd = parselmouth.Sound(query)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']

        mel_time = []
        f0 = []
        
        for i in range(len(pitch_values)):
            f0.append(pitch_values[i].round(decimals=3))
            mel_time.append(pitch.xs()[i].round(decimals=2))
        
        pd.DataFrame({'time':mel_time, 'f0':f0}).to_csv(output_file, index=False)
        end = time.time()
        totaltime += round(end-start, 3)
    avg_time = totaltime/len(querylist)    
f.close()

print("Total time for melody extraction is {:.3f} sec".format(totaltime))
print("Avg time for melody extraction is {:.3f} sec".format(avg_time))