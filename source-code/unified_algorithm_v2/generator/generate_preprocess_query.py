import sys
import glob
import pickle
import pyprind
import pandas as pd
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')

from unified_algorithm_v2.preprocessing import preprocess_query, get_MNF

ioacas_notelist  = []
mirqbsh_notelist = []

ioacas = 'E:\Kuliah\S2\Thesis\source code thesis\database\IOACAS_QBH\midi_note\*.csv'
mirqbsh = 'E:\Kuliah\S2\Thesis\source code thesis\database\MIR-QBSH\midi_note\*.csv'

for f in glob.glob(ioacas):
    ioacas_notelist.append(f)

for f in glob.glob(mirqbsh):
    mirqbsh_notelist.append(f)

def gen_preprocess_query(ouput_path, dataset, n, c):
    for x in n:
        for y in c:
            for z in dataset:
                if z == 'ioacas':
                    notelist = ioacas_notelist
                else :
                    notelist = mirqbsh_notelist
                type = str(z) + '_n' + str(x)
                if y == 1 :
                    type = type + 'c'
                    
                preprocessed_queries = {}
                mnf_feature = {}
                
                bar = pyprind.ProgBar(len(notelist), track_time=True, title='generating {} preprocess query'.format(type))
                for query in notelist:
                    pitch = pd.read_csv(query)
                    note_list = pitch['semitone'].to_numpy() # start from 0.02 same like the extraction result, 2:800 to get 7.99 sec
                    
                    note_list = preprocess_query(note_list, consecutive = x, compressed = y, query_type = 'midi')
                    preprocessed_queries[query] = note_list
                    
                    mnf_midi_query = get_MNF(note_list)
                    mnf_midi_query = ''.join(mnf_midi_query)
                    mnf_feature[query] = mnf_midi_query
                    bar.update()
                
                filename_pq  = str(ouput_path+type+'_pq.pkl')
                filename_mnf = str(ouput_path+type+'_mnf.pkl')
                
                print('saving precomputed feature......', end='\r')
                f = open(filename_pq,"wb")
                pickle.dump(preprocessed_queries, f)
                f.close()
                
                f = open(filename_mnf,"wb")
                pickle.dump(mnf_feature, f)
                f.close()

n = [1, 2]
c = [0, 1]
path = 'E:\Kuliah\S2\Thesis\source code thesis\database\preprocess_query\\'
dataset = ['ioacas', 'mirqbsh']
gen_preprocess_query(path, dataset, n, c)

