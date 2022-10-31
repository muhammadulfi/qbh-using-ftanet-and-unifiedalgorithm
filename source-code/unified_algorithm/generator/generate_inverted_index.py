import sys
import glob
import pickle
import pyprind
import pandas as pd
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')

from unified_algorithm.preprocess import generate_inverted_index, preprocess_query, get_MNF

ioacas_notelist  = []
mirqbsh_notelist = []

ioacas = 'E:\Kuliah\S2\Thesis\source code thesis\database\IOACAS_QBH\midi_note\*.csv'
mirqbsh = 'E:\Kuliah\S2\Thesis\source code thesis\database\MIR-QBSH\midi_note\*.csv'

for f in glob.glob(ioacas):
    ioacas_notelist.append(f)

for f in glob.glob(mirqbsh):
    mirqbsh_notelist.append(f)

def gen_invertedIdx_feature(ouput_path, dataset, n, c):
    print('\nGENERATING INVERTED INDEX...')
    for x in n:
        for y in c:
            type = 'n'+str(x)
            if y == 1 :
                type = 'n'+str(x)+'c'
            for z in dataset:
                if z == 'ioacas':
                    notelist = ioacas_notelist
                else :
                    notelist = mirqbsh_notelist
                type = 'n'+str(x)
                if y == 1 :
                    type = 'n'+str(x)+'c'
                rp2g, rp3g, rp4g = generate_inverted_index(notelist, consecutive = x, compressed = y)
                
                filename_rp2g = str(ouput_path+z+'_'+type+'_rp2g.pkl')
                filename_rp3g = str(ouput_path+z+'_'+type+'_rp3g.pkl')
                filename_rp4g = str(ouput_path+z+'_'+type+'_rp4g.pkl')
                
                print('saving precomputed feature......', end='\r')
                f = open(filename_rp2g,"wb")
                pickle.dump(rp2g,f)
                f = open(filename_rp3g,"wb")
                pickle.dump(rp3g,f)
                f = open(filename_rp4g,"wb")
                pickle.dump(rp4g,f)
                f.close()

def gen_mnf_feature(ouput_path, dataset, n, c):
    print('\nGENERATING MNF FEATURE...')
    for x in n:
        for y in c:
            type = 'n'+str(x)
            if y == 1 :
                type = 'n'+str(x)+'c'
            for z in dataset:
                if z == 'ioacas':
                    notelist = ioacas_notelist
                else :
                    notelist = mirqbsh_notelist
                
                filename = str(ouput_path+z+'_'+type+'_mnf.pkl')
                mnf_feature = {}
                bar = pyprind.ProgBar(len(notelist), track_time=True, title='generating {} mnf feature'.format(type))
                for note in notelist:
                    pitch = pd.read_csv(note)
                    note_list = pitch['semitone'].to_numpy() #2:800 to get 7.99 sec
                    
                    query_midi = preprocess_query(note_list, consecutive = x, compressed = y, type = 'midi')
                    mnf_midi_query = get_MNF(query_midi)
                    mnf_midi_query = ''.join(mnf_midi_query)
                    mnf_feature[note] = mnf_midi_query
                    bar.update()
                
                print('saving precomputed feature......', end='\r')
                f = open(filename,"wb")
                pickle.dump(mnf_feature,f)
                f.close()
    return 0


n = [5,6,7,8,9, 10]
c = [0,1]
# path = 'E:\Kuliah\S2\Thesis\source code thesis\database\inverted_index_short\\'
# path = 'E:\Kuliah\S2\Thesis\source code thesis\database\mnf_feature_short\\'
path = 'E:\Kuliah\S2\Thesis\source code thesis\database\precomputed_feature_v2\\'
dataset = ['ioacas', 'mirqbsh']
gen_invertedIdx_feature(path, dataset, n, c)
gen_mnf_feature(path, dataset, n, c)