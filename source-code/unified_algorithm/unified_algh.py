import math
from unified_algorithm.preprocess import preprocess_query, get_relative_pitch, get_n_gram, get_MNF, generate_ngram_occurance
def get_MRR(rank_lists, n):
    try:
        sigma_all = 0
        for rank in rank_lists:
            if isinstance(rank, int):
                sigma_all += 1/rank
        
        return round(1/abs(n) * sigma_all, 2)
        # return round(sigma_all/n,2)
    except:
        print(rank_lists)
        return 0

def get_top_n_hit_ratio(rank_list, n=10):
    count = 0
    for rank in rank_list:
        if rank <= n:
            count += 1
    
    return round(count/len(rank_list), 2)

def generate_single_inverted_index(rpng):
    from operator import itemgetter
    
    inverted_index = {}
    unique_rpng = sorted(set(rpng))
    frequency_rpng = [rpng.count(x) for x in unique_rpng]
    
    for i in range(len(unique_rpng)):
        inverted_index[unique_rpng[i]] = frequency_rpng[i]
    
    inverted_index = {k: v for k, v in sorted(inverted_index.items(), key=lambda item: item[1], reverse=True)}
    
    return inverted_index

def get_rpng_result(inverted_rpng, inverted_rpng_query):
    rpng_result = {}
    for k, v in inverted_rpng_query.items():
        if k in inverted_rpng:
            for note in inverted_rpng[k]:
                if note[0] not in rpng_result:
                    rpng_result[note[0]] = 0
                if note[0] in rpng_result:
                    count = min(v, note[1])
                    rpng_result[note[0]] += count
    
    # sort by count
    rpng_result = {k: v for k, v in sorted(rpng_result.items(), key=lambda item: item[1], reverse=True)}
    return rpng_result

def get_rpng_result_mod(inverted_rpng, inverted_rpng_query):
    rpng_result = {}
    for k, v in inverted_rpng_query.items():
        if k in inverted_rpng:
            for note in inverted_rpng[k]:
                if note[0] not in rpng_result:
                    rpng_result[note[0]] = 0
                if note[0] in rpng_result:
                    count = min(v, note[1])
                    rpng_result[note[0]] += count
    
    # sort by count
    rpng_result = {k: v for k, v in sorted(rpng_result.items(), key=lambda item: item[1], reverse=True)}
    result = get_rank_mod(rpng_result)
    return result

def get_rank_mod(result):
    res = {}
    i = 1
    v_temp, r_temp = 999, 0
    for k, v in result.items():
        if v_temp == 999 or r_temp == 0 or v_temp != v:
            res[k] =[v,i]
            v_temp = v
            r_temp = i
        else:
            res[k] =[v_temp, r_temp]        
        i+=1
    
    return res

def check_result(top_n_result, best_rank):
    return 0


def unified_algorithm(query, truth, consecutive = 10, compressed = 0, n = 0, dataset = 'mirqbsh', best = 0, smoothing = False):
    import pickle
    import glob
    import itertools
    import pandas as pd
    import editdistance
    
    best_rank = 9999999
    best_algh = ''
    best_result = {}
    
    # prepare path for load inverted index
    precomputed_feature = 'E:\Kuliah\S2\Thesis\source code thesis\database\precomputed_feature_v2\\' # full     
    # inverted_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\short_feature\\' # 8 sec only
    type = 'n'+str(consecutive)
    if compressed :
        type = type+'c'
    filename_rp2g = str(precomputed_feature+dataset+'_'+type+'_rp2g.pkl')
    filename_rp3g = str(precomputed_feature+dataset+'_'+type+'_rp3g.pkl')
    filename_rp4g = str(precomputed_feature+dataset+'_'+type+'_rp4g.pkl')
    filename_mnf  = str(precomputed_feature+dataset+'_'+type+'_mnf.pkl')
    
    # process query
    query = preprocess_query(query, consecutive, compressed, smoothing = smoothing)
    if not query:
        return {}, 'query is null', 0
    
    relative_pitch = get_relative_pitch(query)
    
    # start alghorithm from rp4g
    if best == 0:
        n = 10 # number to check from top n list
    
    # rp4g
    f = open(filename_rp4g,"rb")
    inverted_rp4g = pickle.load(f)
    f.close()
    
    rp4g = get_n_gram(relative_pitch, 3)
    invetred_rp4g_query = generate_single_inverted_index(rp4g)
    
    rp4g_result = get_rpng_result(inverted_rp4g, invetred_rp4g_query)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(rp4g_result.items(), n))
    else :
        top_n_result = rp4g_result
    
    if truth in top_n_result and best == 0:
        list_result = list(top_n_result)
        return rp4g_result, 'rp4g', list_result.index(truth)+1
    elif truth in rp4g_result:
        list_result = list(rp4g_result)
        if min(best_rank, list_result.index(truth)+1) == list_result.index(truth)+1:
            best_result = rp4g_result
            best_rank = list_result.index(truth)+1
            best_algh = 'rp4g'
    
    # rp3g 
    f = open(filename_rp3g,"rb")
    inverted_rp3g = pickle.load(f)
    f.close()
    
    rp3g = get_n_gram(relative_pitch, 2)
    invetred_rp3g_query = generate_single_inverted_index(rp3g)
    
    rp3g_result = get_rpng_result(inverted_rp3g, invetred_rp3g_query)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(rp3g_result.items(), n))
    else :
        top_n_result = rp3g_result
        
    if truth in top_n_result and best == 0:
        list_result = list(top_n_result)
        return rp3g_result, 'rp3g', list_result.index(truth)+1
    elif truth in rp3g_result:
        list_result = list(rp3g_result)
        if min(best_rank, list_result.index(truth)+1) == list_result.index(truth)+1:
            best_result = rp3g_result
            best_rank = list_result.index(truth)+1
            best_algh = 'rp3g'
    
    # rp2g
    f = open(filename_rp2g,"rb")
    inverted_rp2g = pickle.load(f)
    f.close()
    
    rp2g = get_n_gram(relative_pitch, 1)
    invetred_rp2g_query = generate_single_inverted_index(rp2g)
    
    rp2g_result = get_rpng_result(inverted_rp2g, invetred_rp2g_query)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(rp2g_result.items(), n))
    else :
        top_n_result = rp2g_result
        
    
    if truth in top_n_result and best == 0:
        list_result = list(top_n_result)
        return rp2g_result, 'rp2g', list_result.index(truth)+1
    elif truth in rp2g_result:
        list_result = list(rp2g_result)
        if min(best_rank, list_result.index(truth)+1) == list_result.index(truth)+1:
            best_result = rp2g_result
            best_rank = list_result.index(truth)+1
            best_algh = 'rp2g'
    
    # mnf
    f = open(filename_mnf,"rb")
    mnf_feature = pickle.load(f)
    f.close()
    
    # mnf query
    mnf_query = get_MNF(query)
    mnf_query = ''.join(mnf_query)
    mnf_result = {}
    for note, feature in mnf_feature.items():
        mnf_result[note] = editdistance.eval(feature, mnf_query)
    
    # sort by count
    mnf_result = {k: v for k, v in sorted(mnf_result.items(), key=lambda item: item[1])}
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(mnf_result.items(), n))
    else :
        top_n_result = mnf_result
        
    #
    if truth in top_n_result and best == 0:
        list_result = list(top_n_result)
        return mnf_result, 'mnf', list_result.index(truth)+1
    elif truth in mnf_result:
        list_result = list(mnf_result)
        if min(best_rank, list_result.index(truth)+1) == list_result.index(truth)+1:
            best_result = mnf_result
            best_rank = list_result.index(truth)+1
            best_algh = 'mnf'
    if best_result:
        return best_result, best_algh, best_rank
    else:
        return {}, 'not found', 0

def unified_algorithm_mod(query, truth, consecutive = 10, compressed = 0, n = 0, dataset = 'mirqbsh', best = 0, smoothing = False):
    import pickle
    import glob
    import itertools
    import pandas as pd
    import editdistance
    
    best_rank = 9999999
    best_algh = ''
    best_result = {}
    
    # prepare path for load inverted index
    precomputed_feature = 'E:\Kuliah\S2\Thesis\source code thesis\database\precomputed_feature_v2\\' # full     
    # inverted_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\short_feature\\' # 8 sec only
    type = 'n'+str(consecutive)
    if compressed :
        type = type+'c'
    filename_rp2g = str(precomputed_feature+dataset+'_'+type+'_rp2g.pkl')
    filename_rp3g = str(precomputed_feature+dataset+'_'+type+'_rp3g.pkl')
    filename_rp4g = str(precomputed_feature+dataset+'_'+type+'_rp4g.pkl')
    filename_mnf  = str(precomputed_feature+dataset+'_'+type+'_mnf.pkl')
    
    # process query
    query = preprocess_query(query, consecutive, compressed, smoothing = smoothing)
    if not query:
        return {}, 'query is null', 0
    
    relative_pitch = get_relative_pitch(query)
    
    # start alghorithm from rp4g
    if best == 0:
        n = 10 # number to check from top n list
    
    # rp4g
    f = open(filename_rp4g,"rb")
    inverted_rp4g = pickle.load(f)
    f.close()
    
    rp4g = get_n_gram(relative_pitch, 3)
    invetred_rp4g_query = generate_single_inverted_index(rp4g)
    
    rp4g_result = get_rpng_result_mod(inverted_rp4g, invetred_rp4g_query)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(rp4g_result.items(), n))
    else :
        top_n_result = rp4g_result
    
    if truth in top_n_result and best == 0:
        return rp4g_result, 'rp4g', rp4g_result[truth][1]
    elif truth in rp4g_result:
        if min(best_rank, rp4g_result[truth][1]) == rp4g_result[truth][1]:
            best_result = rp4g_result
            best_rank = rp4g_result[truth][1]
            best_algh = 'rp4g'
    
    # rp3g 
    f = open(filename_rp3g,"rb")
    inverted_rp3g = pickle.load(f)
    f.close()
    
    rp3g = get_n_gram(relative_pitch, 2)
    invetred_rp3g_query = generate_single_inverted_index(rp3g)
    
    rp3g_result = get_rpng_result_mod(inverted_rp3g, invetred_rp3g_query)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(rp3g_result.items(), n))
    else :
        top_n_result = rp3g_result
        
    if truth in top_n_result and best == 0:
        return rp3g_result, 'rp3g', rp3g_result[truth][1]
    elif truth in rp3g_result:
        if min(best_rank, rp3g_result[truth][1]) == rp3g_result[truth][1]:
            best_result = rp3g_result
            best_rank = rp3g_result[truth][1]
            best_algh = 'rp3g'
    
    # rp2g
    f = open(filename_rp2g,"rb")
    inverted_rp2g = pickle.load(f)
    f.close()
    
    rp2g = get_n_gram(relative_pitch, 1)
    invetred_rp2g_query = generate_single_inverted_index(rp2g)
    
    rp2g_result = get_rpng_result_mod(inverted_rp2g, invetred_rp2g_query)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(rp2g_result.items(), n))
    else :
        top_n_result = rp2g_result
        
    
    if truth in top_n_result and best == 0:
        list_result = list(top_n_result)
        return rp2g_result, 'rp2g', rp2g_result[truth][1]
    elif truth in rp2g_result:
        list_result = list(rp2g_result)
        if min(best_rank, rp2g_result[truth][1]) == rp2g_result[truth][1]:
            best_result = rp2g_result
            best_rank = rp2g_result[truth][1]
            best_algh = 'rp2g'
    
    # mnf
    f = open(filename_mnf,"rb")
    mnf_feature = pickle.load(f)
    f.close()
    
    # mnf query
    mnf_query = get_MNF(query)
    mnf_query = ''.join(mnf_query)
    mnf_result = {}
    for note, feature in mnf_feature.items():
        mnf_result[note] = editdistance.eval(feature, mnf_query)
    
    # sort by count
    mnf_result = {k: v for k, v in sorted(mnf_result.items(), key=lambda item: item[1])}
    
    mnf_result = get_rank_mod(mnf_result)
    
    # get top n from result
    if n != 0: 
        top_n_result = dict(itertools.islice(mnf_result.items(), n))
    else :
        top_n_result = mnf_result
        
    #
    if truth in top_n_result and best == 0:
        list_result = list(top_n_result)
        return mnf_result, 'mnf', list_result.index(truth)+1
    elif truth in mnf_result:
        list_result = list(mnf_result)
        if min(best_rank, list_result.index(truth)+1) == list_result.index(truth)+1:
            best_result = mnf_result
            best_rank = list_result.index(truth)+1
            best_algh = 'mnf'
    if best_result:
        return best_result, best_algh, best_rank
    else:
        return {}, 'not found', 0