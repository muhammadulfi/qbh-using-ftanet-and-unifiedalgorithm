import math
import pickle

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

def generate_single_inverted_index(rpng):
    inverted_index = {}
    unique_rpng = sorted(set(rpng))
    frequency_rpng = [rpng.count(x) for x in unique_rpng]
    
    for i in range(len(unique_rpng)):
        inverted_index[unique_rpng[i]] = frequency_rpng[i]
    
    inverted_index = {k: v for k, v in sorted(inverted_index.items(), key=lambda item: item[1], reverse=True)}
    
    return inverted_index

def preprocess_query(query, consecutive = 10, compressed = 0, query_type = 'query', smoothing = False, thres = 30):
    if query_type == 'query':
        if smoothing:
            query = pitch_smoothing(query, thres)
        query = f0_to_semitone(query)
    query = get_n_consecutive_pitch(query, consecutive)
    
    if compressed:
        query = get_compressed_query(query)
    
    return query
