import numpy as np 

ioacas_path  = 'E:\Kuliah\S2\Thesis\source code thesis\database\IOACAS_QBH\\'
mirqbsh_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\MIR-QBSH\\'

ioacas_truth = 'E:\Kuliah\S2\Thesis\source code thesis\database\IOACAS_QBH\query.list'
mirqbsh_wav = 'E:\Kuliah\S2\Thesis\source code thesis\database\MIR-QBSH\querywav_list.txt'

ioacas_csv  = ''
mirqbsh_csv = ''
# need to add csv in list

ioacas_truthlist  = []
mirqbsh_truthlist = []
# ioacas
with open(ioacas_truth,'r') as f:
    ioacas_truth = f.readlines()
    # wav_path = str(ioacas_path + 'wavfile\\')
    mid_path = str(ioacas_path + 'midi_note\\')
    for row in ioacas_truth:
        row = row.replace('\n', '')
        row = row.split('\t')
        ioacas_truthlist.append(str(ioacas_path+row[0] +',' + mid_path+row[1]+'.csv'))

with open(mirqbsh_wav,'r') as f:
    mirqbsh_wav = f.readlines()
    for row in mirqbsh_wav:
        row = row.replace('\n', '')
        splitted_row = row.split('\\')
        midfile = splitted_row[-1]
        midfile = midfile.replace('.wav','.csv')
        path = str(mirqbsh_path) +'midi_note\\'
        mirqbsh_truthlist.append(str(row +','+path+midfile))

f.close()

with open(str(ioacas_path+"groundtruth.txt"), 'w') as output:
    for row in ioacas_truthlist:
        output.write(str(row) + '\n')
output.close()

with open(str(mirqbsh_path+"groundtruth.txt"), 'w') as output:
    for row in mirqbsh_truthlist:
        output.write(str(row) + '\n')
output.close()