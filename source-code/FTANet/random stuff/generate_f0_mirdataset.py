import numpy as np
import pandas as pd
import os
import time
from cfp import getTimeArr
from loader import semitone_to_f0

with open(".\database\MIR-QBSH\querywav_list.txt",'r') as f:
    querylist = f.readlines()
    

start = time.time()
for row in querylist: 
    # prepare the output file
    file = row.split("\\")
    filename = '-'.join(file[-3:])
    output_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\MIR-QBSH\\f0File', filename + ".csv")
    output_file = output_file.replace(".wav","")
    output_file = output_file.replace("\n","")
    
    # prepare query path 
    row = row.replace("\n","")
    pvpath = row.replace(".wav",".pv")
    
    pv = np.loadtxt(pvpath)
    pitch = np.array(semitone_to_f0(pv))
    time_arr = getTimeArr(row, model_type='vocal', sr=8000, hop=256, window=768)
    # ref_arr = np.column_stack((time_arr, pitch))
    print(time_arr)
    print(len(time_arr))
    
    pd.DataFrame({'time':time_arr, 'f0':pitch[1:]}).to_csv(output_file, index=False)
    print('Done generate f0 for ', output_file)

end = time.time()
print('Done in ', round(end-start, 3),' sec or ', round(end-start, 3)/60,' min')