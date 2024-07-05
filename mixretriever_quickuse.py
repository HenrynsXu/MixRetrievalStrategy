'''
Quick deployment for our strategy when only passage id and first-stage retrieval scores are available.
'''

import numpy as np
import pandas as pd

import jieba

import spacy
en_lang = spacy.load('en_core_web_sm')


def linear_norm(x):
    minx,maxx = min(x),max(x)
    for i in range(len(x)):
        x[i] = (x[i]-minx)/(maxx-minx+1e-5)
    return x


def quick_search(query, 
                 all_results_jsonl, 
                 idf_dictionary:dict, 
                 lang = 'zh'):
    '''
    `all_results_jsonl` is the file where each line contains following JSON object.
    {'id': str, 'sparse_score': float, 'dense_score': float, 'fine_grain_scores'(optional): List[float]}
    '''
    df = pd.read_json(all_results_jsonl,lines=True)
    if 'fine_grain_scores' in df.columns:
        if df['fine_grain_scores'].isna().any():
            use_mul_grain = False
        else:
            use_mul_grain = True
    else:
        use_mul_grain = False

    if lang == 'zh': 
        query_l=jieba.lcut(query)
    else: 
        query_l = []
        for w in en_lang(query):  
            query_l.append(w.text)
    idf_dic = idf_dictionary
    values = list(idf_dic.values())
    values.sort(reverse=True)

    if use_mul_grain:
        mulgrain_k = 5
        stc_sim = df['fine_grain_scores'].apply(lambda x: max(x))
        df['stc_sim'] = stc_sim
        df['range'] = df['fine_grain_scores'].apply(lambda x: max(x) - min(x))
        psg_res = df.sort_values("dense_score", ascending = False, ignore_index = True).head(mulgrain_k)
        stc_res = df.sort_values("stc_sim", ascending = False, ignore_index = True).head(mulgrain_k)
        p_range = np.mean(psg_res['range'])
        s_range = np.mean(stc_res['range'])
        if s_range-p_range < 1e-6 or s_range-p_range > 0.15:
            weights = [20,80]
        else:
            weights = [60,40]
        
        df['multi_score'] = np.average([df.psg_sim.tolist(),df.stc_sim.tolist()],weights = weights,axis = 0)
        final_embed_score = df['multi_score']
    else:
        final_embed_score = df['dense_score']

    query_idfs = [idf_dic.get(q,0) for q in query_l]
    query_idfs.sort(reverse=True)

    thres = 0.6
    weight = 0.7

    top1_thres = values[int((thres-0.1)*len(values))]
    topk_thres = values[int(thres*len(values))]
    if query_idfs[0]>top1_thres and np.mean(query_idfs[:3]) > topk_thres:
        df['mix_score'] = np.average([linear_norm(df.sparse_score.tolist()),linear_norm(final_embed_score.tolist())],axis=0,weights=[weight,1-weight])
    else:
        df['mix_score'] = np.average([linear_norm(df.sparse_score.tolist()),linear_norm(final_embed_score.tolist())],axis=0,weights=[1-weight,weight])
    df.sort_values("mix_score", ascending=False, ignore_index=True,inplace=True)
    df = df[['id','final_score']]
    df.to_json('output.jsonl',orient='records',lines=True)
    