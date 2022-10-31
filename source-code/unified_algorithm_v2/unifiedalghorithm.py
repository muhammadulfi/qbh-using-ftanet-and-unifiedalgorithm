import os
import sys
import math
import glob
import time
import pickle
import pyprind
import datetime
import itertools
import editdistance
import numpy as np
import pandas as pd
sys.path.insert(0, 'E:\\Kuliah\\S2\\Thesis\\source code thesis')

from unified_algorithm_v2.preprocessing import preprocess_query, get_relative_pitch, get_n_gram, get_MNF, generate_single_inverted_index

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

def rpng_matching(query, ref, n, query_len, slide = 3):
    result = {}
    for ref_name, feature in ref.items():
        i = 0
        best = 0
        while i+query_len <= len(feature)+query_len:
            relative_pitch_ref = get_relative_pitch(feature[i:i+query_len])  # either len(query) or time duration
            rpng_ref = get_n_gram(relative_pitch_ref, n)
            invetred_rpng_ref = generate_single_inverted_index(rpng_ref)
            counted = 0
            for note, count in query.items():
                if note in invetred_rpng_ref:
                    counted += min(count, invetred_rpng_ref[note])
            best = max(best, counted)
            i += slide
        
        result[ref_name] = best
        
    result_sorted = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
    return result_sorted

def mnf_mathing(query, ref, slide = 3):
    result = {}
    for ref_name, feature in ref.items():
        i = 0
        best = 99999999
        while i+len(query) <= len(feature)+len(query):
            distance = editdistance.eval(query, feature[i:i+len(query)])            
            best = min(best, distance)
            i += slide
        
        result[ref_name] = best
    
    result_sorted = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
    return result_sorted

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

def unified_algorithm(query, truth, consecutive = 10, compressed = 0, n = 0, dataset = 'mirqbsh', best = 0, smoothing = False):
    best_rank = 9999999
    best_algh = ''
    best_result = {}
    preprocess_query_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\preprocess_query\\'
    type = 'n'+str(consecutive)
    if compressed :
        type = type+'c'
    filename_pq = str(preprocess_query_path+dataset+'_'+type+'_pq.pkl')
    filename_mnf  = str(preprocess_query_path+dataset+'_'+type+'_mnf.pkl')
    
    # process query
    query = preprocess_query(query, consecutive, compressed, smoothing = smoothing)
    if not query:
        return {}, 'query is null', 0
    
    relative_pitch = get_relative_pitch(query)
    
    # start alghorithm from rp4g
    if best == 0:
        n = 10 # number to check from top n list
    
    # preprocess query load
    f = open(filename_pq,"rb")
    preprocessed_query_ref = pickle.load(f)
    f.close()
    
    rp4g = get_n_gram(relative_pitch, 3)
    invetred_rp4g_query = generate_single_inverted_index(rp4g)
    
    rp4g_result = rpng_matching(invetred_rp4g_query, preprocessed_query_ref, 3, len(query))
    
    # print(rp4g_result)
    
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
    
    rp3g = get_n_gram(relative_pitch, 2)
    invetred_rp3g_query = generate_single_inverted_index(rp3g)
    
    rp3g_result = rpng_matching(invetred_rp3g_query, preprocessed_query_ref, 2, len(query))
    
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
    
    rp2g = get_n_gram(relative_pitch, 1)
    invetred_rp2g_query = generate_single_inverted_index(rp2g)
    
    rp2g_result = rpng_matching(invetred_rp2g_query, preprocessed_query_ref, 1, len(query))
    
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
    
    f = open(filename_mnf,"rb")
    preprocessed_mnf_ref = pickle.load(f)
    f.close()
    
    mnf_query = get_MNF(query)
    mnf_query = ''.join(mnf_query)
    
    mnf_result = mnf_mathing(mnf_query, preprocessed_mnf_ref)
    
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
    best_rank = 9999999
    best_algh = ''
    best_result = {}
    preprocess_query_path = 'E:\Kuliah\S2\Thesis\source code thesis\database\preprocess_query\\'
    type = 'n'+str(consecutive)
    if compressed :
        type = type+'c'
    filename_pq = str(preprocess_query_path+dataset+'_'+type+'_pq.pkl')
    filename_mnf  = str(preprocess_query_path+dataset+'_'+type+'_mnf.pkl')
    
    # process query
    query = preprocess_query(query, consecutive, compressed, smoothing = smoothing)
    if not query:
        return {}, 'query is null', 0
    
    relative_pitch = get_relative_pitch(query)
    
    # start alghorithm from rp4g
    if best == 0:
        n = 10 # number to check from top n list
    
    # preprocess query load
    f = open(filename_pq,"rb")
    preprocessed_query_ref = pickle.load(f)
    f.close()
    
    rp4g = get_n_gram(relative_pitch, 3)
    invetred_rp4g_query = generate_single_inverted_index(rp4g)
    
    rp4g_result = rpng_matching(invetred_rp4g_query, preprocessed_query_ref, 3, len(query))
    
    rp4g_result = get_rank_mod(rp4g_result)
    
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
    
    rp3g = get_n_gram(relative_pitch, 2)
    invetred_rp3g_query = generate_single_inverted_index(rp3g)
    
    rp3g_result = rpng_matching(invetred_rp3g_query, preprocessed_query_ref, 2, len(query))
    
    rp3g_result = get_rank_mod(rp3g_result)
    
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
    
    rp2g = get_n_gram(relative_pitch, 1)
    invetred_rp2g_query = generate_single_inverted_index(rp2g)
    
    rp2g_result = rpng_matching(invetred_rp2g_query, preprocessed_query_ref, 1, len(query))
    
    rp2g_result = get_rank_mod(rp2g_result)
    
    if n != 0: 
        top_n_result = dict(itertools.islice(rp2g_result.items(), n))
    else :
        top_n_result = rp2g_result
        
    
    if truth in top_n_result and best == 0:
        return rp2g_result, 'rp2g', rp2g_result[truth][1]
    elif truth in rp2g_result:
        if min(best_rank, rp2g_result[truth][1]) == rp2g_result[truth][1]:
            best_result = rp2g_result
            best_rank = rp2g_result[truth][1]
            best_algh = 'rp2g'
    
    f = open(filename_mnf,"rb")
    preprocessed_mnf_ref = pickle.load(f)
    f.close()
    
    mnf_query = get_MNF(query)
    mnf_query = ''.join(mnf_query)
    
    mnf_result = mnf_mathing(mnf_query, preprocessed_mnf_ref)
    
    mnf_result = get_rank_mod(mnf_result)

    if n != 0: 
        top_n_result = dict(itertools.islice(mnf_result.items(), n))
    else :
        top_n_result = mnf_result
    #
    if truth in top_n_result and best == 0:
        return mnf_result, 'mnf', mnf_result[truth][1]
    elif truth in mnf_result:
        if min(best_rank, mnf_result[truth][1]) == mnf_result[truth][1]:
            best_result = mnf_result
            best_rank = mnf_result[truth][1]
            best_algh = 'mnf'
    if best_result:
        return best_result, best_algh, best_rank
    else:
        return {}, 'not found', 0


def run_experiments():
    datasets = ['ioacas']
    n = [10]
    c = [0, 1]
    best = 0
    fta_result = {}
    time_computation = {}
    alg_experiment = {}
    unified_result = {}
    smoothing_pitch = False
    extraction = 'rc'

    log_filename = 'fix_qlen_log_10c_i_'+extraction
    if best == 1:
        log_filename = log_filename+'_best'
    else:
        log_filename = log_filename+'_notbest'

    if smoothing_pitch:
        log_filename = log_filename+'_smooth'
    else:
        log_filename = log_filename+'_notsmooth'

    log_unified = open('./log/retrieval_submatching/{}.txt'.format(log_filename), 'w')

    for dataset in datasets:
        if dataset == 'mir-qbsh':
            groundtruth_list = 'E:\Kuliah\S2\Thesis\source code thesis\database\\'+'MIR-QBSH\\'+'groundtruth.txt'
        else :
            groundtruth_list = 'E:\Kuliah\S2\Thesis\source code thesis\database\\'+'IOACAS_QBH\\'+'groundtruth.txt'

        groundtruthes = np.loadtxt(groundtruth_list, delimiter = ",", dtype = 'str')
        query_count = len(groundtruthes)
        for x in n:
            for y in c:
                top1, top3, top5, top10, mrr = 0, 0, 0, 0, 0
                alg_result = []
                rank_list = []
                experiment_result = []
                count_null = 0
                total_time = 0
                type = dataset +'_'+ str(x) +'n'
                if y == 1:
                    type = type + 'c'
                
                if y == 1 :
                    compressed_status = 'compressed'
                else:
                    compressed_status = 'uncompressed'
                beginprocess = datetime.datetime.now()
                bar = pyprind.ProgPercent(query_count, track_time=True, stream=1, title='{} is start at {}'.format(type, beginprocess))
                # print('{} is start'.format(type), end='\r')
                for groundtruth in groundtruthes:
                    query = groundtruth[0]
                    truth = groundtruth[1]

                    # load extracted wav
                    query = query.replace("\n","")
                    # prepare the output file
                    file = query.split("\\")
                    if dataset == 'mir-qbsh':
                        filename = '-'.join(file[-3:])
                    else:
                        filename = file[-1]
                    
                    if not extraction == 'praat':
                        query_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\result_ftanet\\rc\\'+ dataset +'\\'+ filename + ".csv")
                        experiment_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\var_submatching\\ftanet_rc\\fix_qlen_'+ type + ".pkl") 
                    else:
                        query_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\result_praat\\'+ dataset +'\\'+ filename + ".csv")
                        experiment_file = os.path.join('E:\Kuliah\S2\Thesis\source code thesis\database\\var_submatching\\praat\\fix_qlen_'+ type + ".pkl") 
                    query_file = query_file.replace(".wav","")
                    pitch = pd.read_csv(query_file)
                    f0 = pitch['f0'].to_numpy()
                    
                    if best == 0 :
                        experiment_file = experiment_file.replace('.pkl', '_notbest.pkl')
                    if smoothing_pitch:
                        experiment_file = experiment_file.replace('.pkl', '_smoothing.pkl')
                    start = time.time()
                    if dataset == 'ioacas':
                        result, alg, rank = unified_algorithm(f0, truth, consecutive = x, compressed = y, dataset = 'ioacas', best = best, smoothing = smoothing_pitch)
                    else:
                        result, alg, rank = unified_algorithm(f0, truth, consecutive = x, compressed = y, best = best, smoothing = smoothing_pitch)
                    
                    if alg != 'not found' and alg != 'query is null':
                        rank_list.append(rank)
                        if rank <= 1:
                            top1+=1
                            top3+=1
                            top5+=1
                            top10+=1
                        elif rank <= 3:
                            top3+=1
                            top5+=1
                            top10+=1
                        elif rank <= 5:
                            top5+=1
                            top10+=1
                        elif rank <= 10:
                            top10+=1
                        
                    else : 
                        # print(rank)
                        rank = len(groundtruthes)
                        rank_list.append(rank)
                        
                    if alg == 'query is null':
                        count_null += 1
                        log_unified.write('{}, {}\n'.format(query_file, type))                    
                    
                    alg_result.append(alg)
                    experiment_result.append([query, alg, rank])
                    end = time.time()
                    total_time += round(end-start, 3)
                    bar.update()
                    
                avg_time = total_time/len(alg_result) 
                
                top1_hit_ratio = top1/len(alg_result)
                top3_hit_ratio = top3/len(alg_result)
                top5_hit_ratio = top5/len(alg_result)
                top10_hit_ratio = top10/len(alg_result)
                
                mrr = get_MRR(rank_lists = rank_list, n = len(alg_result))
                
                fta_result[type] = [mrr, top1_hit_ratio, top1, top3_hit_ratio, top3, top5_hit_ratio, top5, top10_hit_ratio, top10, total_time, avg_time]
                alg_used = {}
                for alg in alg_result:
                    if alg not in alg_used:
                        alg_used[alg] = 0
                    if alg in alg_used:
                        alg_used[alg] += 1
                
                alg_used = {k: v for k, v in sorted(alg_used.items(), key=lambda item: item[0], reverse=True)}
                alg_experiment[type] = alg_used
                with open(experiment_file, 'wb') as file:
                    pickle.dump(experiment_result, file)

    print('----result----')
    log_unified.write('\n----result----\n')
    print('type\t\t  mrr\ttop1\ttop3\ttop5\ttop10\ttotal time\t\tavg time')
    log_unified.write('type\t: mrr\ttop1\ttop3\ttop5\ttop10\ttotal time\t\tavg time\n')
    for key, value in fta_result.items():
        print('{}\t: {:.4f}\t{:.4f}({})\t{:.4f}({})\t{:.4f}({})\t{:.4f}({})\t{:.4f}\t{:.4f}'.format(
            key, value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9], value[10]))
        log_unified.write('{}\t: {:.4f}\t{:.4f}({})\t{:.4f}({})\t{:.4f}({})\t{:.4f}({})\t{:.4f}\t{:.4f}\n'.format(
            key, value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7], value[8], value[9], value[10]))

    print('type\t\t: alg total')
    log_unified.write('type\t\t: alg total\n')
    for key, values in alg_experiment.items():
        print(key, '\t: ', values)
        log_unified.write('{}\t: {}\n'.format(key, values))