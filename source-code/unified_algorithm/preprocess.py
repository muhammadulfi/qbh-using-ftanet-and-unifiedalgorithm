import math
import pickle

def smoothing_query(query):
    smooth_query = []
    last_query = None
    for f0 in query:
        if f0 == 0:
            smooth_query.append(f0)
        else:
            if last_query is None:
                smooth_query.append(f0)
            else:
                smooth_query.append(f0)
    return 0

def f0_to_semitone(f0):
    semitone = []
    for f in f0:
        if f != 0:
            semitone.append(round(12*math.log((f/440),2)+69))
        else:
            semitone.append(f)
    return semitone

def semitone_to_f0(s):
    f0 = []
    for f in s:
        if f != 0:
            f0.append(round(440*(2**(f/12-23/4)),0))
        else:
            f0.append(f)
    return f0

def get_n_consecutive_pitch(query, n):
    pitch = []
    x=1
    checked_f = None
    for f in query:
        if checked_f is None:
            checked_f = f
            x+=1
        else :
            if checked_f != f or x == n:
                if x == n and checked_f != 0:
                    pitch.append(checked_f)                
                checked_f = f
                x=1
            else :
                x+=1
    return pitch

def get_compressed_query(query):
    compressed_query = []
    for s in query:
        # if s != 0:
            if not compressed_query:
                compressed_query.append(s)
            elif s != compressed_query[-1]:
                compressed_query.append(s)
    return compressed_query

def get_relative_pitch(query):
    relative_pitch = []
    for i in range(len(query)-1):
        relative_pitch.append(query[i+1]-query[i])
    return relative_pitch

def get_n_gram(query, n):
    ngrams = []
    ngrams_list = []
    
    if (n<=1):
        for note in query:
            ngrams_list.append(note)
    else :
        ngrams = zip(*[query[i:] for i in range(n)])
        for ngram in ngrams:
            ngrams_list.append(ngram)
    
    return ngrams_list

def get_MNF(query):
    range_query = []
    mnf = []
    
    counted_note = dict((x,query.count(x)) for x in set(query))
    counted_note = dict(sorted(counted_note.items(), key=lambda item: item[1], reverse=True))
    most_occ_note = list(counted_note.keys())[0]
    
    for note in query:
        range_query.append(note-most_occ_note)
    
    for i in range(len(range_query)):
        mnf.append(chr(int(78+range_query[i]))) # 78 ascii of N
    
    return mnf

def generate_ngram_occurance(rpng):
    # get unique relative pitch and counted rp
    unique_rpng = sorted(set(rpng))
    frequency_rpng = [rpng.count(x) for x in unique_rpng]
    
    ngram_occurance = {}
    
    for i in range(len(unique_rpng)):
        if unique_rpng[i] not in ngram_occurance:
            ngram_occurance[unique_rpng[i]] = []
        if unique_rpng[i] in ngram_occurance:
            ngram_occurance[unique_rpng[i]].append(frequency_rpng[i])
    
    return ngram_occurance

def get_inverted_index(notefile, rpng, inverted_rpng):
    unique_rpng = sorted(set(rpng))
    frequency_rpng = [rpng.count(x) for x in unique_rpng]
    
    for i in range(len(unique_rpng)):
        if unique_rpng[i] not in inverted_rpng:
            inverted_rpng[unique_rpng[i]] = []
        if unique_rpng[i] in inverted_rpng:
            inverted_rpng[unique_rpng[i]].append([notefile, frequency_rpng[i]])
    
    return inverted_rpng

def generate_inverted_index(notes, consecutive = 10, compressed = 0):    
    import pyprind
    import pandas as pd
    from operator import itemgetter
    
    inverted_rp2g = {}
    inverted_rp3g = {}
    inverted_rp4g = {}
    
    bar = pyprind.ProgBar(len(notes), track_time=True, title='generating {}n{} inverted index'.format(consecutive, compressed))
    for note in notes:
        pitch = pd.read_csv(note)
        note_list = pitch['semitone'].to_numpy() # start from 0.02 same like the extraction result, 2:800 to get 7.99 sec
        
        note_list = get_n_consecutive_pitch(note_list, consecutive)
        
        if compressed:
            note_list = get_compressed_query(note_list)
        
        relative_pitch = get_relative_pitch(note_list)
        rp2g = get_n_gram(relative_pitch, 1)
        rp3g = get_n_gram(relative_pitch, 2)
        rp4g = get_n_gram(relative_pitch, 3)
        
        # get unique relative pitch and counted rp
        unique_rp2g = sorted(set(rp2g))
        frequency_rp2g = [rp2g.count(x) for x in unique_rp2g]
        
        unique_rp3g = sorted(set(rp3g))
        frequency_rp3g = [rp3g.count(x) for x in unique_rp3g]
        
        unique_rp4g = sorted(set(rp4g))
        frequency_rp4g = [rp4g.count(x) for x in unique_rp4g]
        
        # add to dict
        for i in range(len(unique_rp2g)):
            if unique_rp2g[i] not in inverted_rp2g:
                inverted_rp2g[unique_rp2g[i]] = []
            if unique_rp2g[i] in inverted_rp2g:
                inverted_rp2g[unique_rp2g[i]].append([note, frequency_rp2g[i]])
        
        for i in range(len(unique_rp3g)):
            if unique_rp3g[i] not in inverted_rp3g:
                inverted_rp3g[unique_rp3g[i]] = []
            if unique_rp3g[i] in inverted_rp3g:
                inverted_rp3g[unique_rp3g[i]].append([note, frequency_rp3g[i]])
        
        for i in range(len(unique_rp4g)):
            if unique_rp4g[i] not in inverted_rp4g:
                inverted_rp4g[unique_rp4g[i]] = []
            if unique_rp4g[i] in inverted_rp4g:
                inverted_rp4g[unique_rp4g[i]].append([note, frequency_rp4g[i]])
        bar.update()
    
    # sort list in each key
    for k, v in inverted_rp2g.items():
        inverted_rp2g[k] = sorted(v, key=itemgetter(1), reverse=True)
    
    for k, v in inverted_rp3g.items():
        inverted_rp3g[k] = sorted(v, key=itemgetter(1), reverse=True)
    
    for k, v in inverted_rp4g.items():
        inverted_rp4g[k] = sorted(v, key=itemgetter(1), reverse=True)
    
    # sort by count in list in each key 
    inverted_rp2g = {key: value for key, value in sorted(inverted_rp2g.items(), key=lambda item: len(item[1]))}
    inverted_rp3g = {key: value for key, value in sorted(inverted_rp3g.items(), key=lambda item: len(item[1]))}
    inverted_rp4g = {key: value for key, value in sorted(inverted_rp4g.items(), key=lambda item: len(item[1]))}
    
    return inverted_rp2g, inverted_rp3g, inverted_rp4g

def pitch_smoothing(pitch, thres = 30):
    smooth_pitch = []
    for i in range(len(pitch)):
        if i != 0 and i != len(pitch)-1:
            if abs(pitch[i-1] - pitch[i]) >= thres or abs(pitch[i+1] - pitch[i]) >= thres:
                if pitch[i+1] != 0 or pitch[i-1] != 0:
                    smooth_pitch.append(pitch[i-1]/2 + pitch[i+1]/2)
                else:
                    smooth_pitch.append(pitch[i])
            else:
                smooth_pitch.append(pitch[i])
        else:
            smooth_pitch.append(pitch[i])
        
    return smooth_pitch

def preprocess_query(query, consecutive = 10, compressed = 0, type = 'query', smoothing = False, thres = 30):
    if type == 'query':
        if smoothing:
            query = pitch_smoothing(query, thres)
        query = f0_to_semitone(query)
    query = get_n_consecutive_pitch(query, consecutive)
    
    if compressed:
        query = get_compressed_query(query)
    
    return query

