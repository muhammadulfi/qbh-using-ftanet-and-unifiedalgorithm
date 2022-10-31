import numpy as np
import pandas as pd
import time
import os
from cfp import cfp_process

# this module is to get feature from QBH dataset for re-training process
totaltime = 0
with open(".\database\MIR-QBSH\querywav_list.txt",'r') as f:
    querylist = f.readlines()
    # small size
    # querylist = querylist[:1000]
    
    for row in querylist:
        start = time.time()
        
        # prepare the output file
        file = row.split("\\")
        filename = '-'.join(file[-3:])
        output_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\cfp_feature_github', filename + ".npy")
        output_file = output_file.replace(".wav","")
        output_file = output_file.replace("\n","")
        
        # prepare query path 
        query = '/'.join(file[-6:])
        query = query.replace("\n","")
        query = './' + query
        
        # start cfp extraction
        data, CenFreq, time_arr = cfp_process(query, model_type='vocal', sr=8000, hop=80, window=768)
        np.save(output_file, data)
        end = time.time()
        totaltime += round(end-start, 3)
    avg_time = totaltime/len(querylist)   
f.close()

print("Total time {:.3f} sec or {:.1f} hours".format(totaltime, totaltime/60))
print("with average {:.3f} sec for each file".format(avg_time))