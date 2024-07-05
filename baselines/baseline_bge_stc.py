import pandas as pd
import numpy as np
import time
from FlagEmbedding import FlagModel


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class BgeStcSearcher:
    '''
    doc_path: a JSONL file, with `id`, `text`, `stc` as keys for each line.
    '''
    def __init__(self, doc_path):
        self.df = self._construct_df(doc_path)
        self.ds_id = doc_path.split('/')[1]
        self.embed_model = FlagModel('bge-large-en-v1.5',query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        self._calculate_psg_embed()
        self._calculate_stc_embed()
    
    def _construct_df(self, doc_path):
        '''
        format: a list of json dicts
        '''
        df = pd.read_json(doc_path)
        assert 'text' in df.columns
        return df
    
    def _calculate_psg_embed(self):
        psg_embed = self.df.text.apply(lambda x: self.embed_model.encode(x))
        self.df['psg_embed'] = psg_embed
    
    def _calculate_stc_embed(self):
        stc_embed = self.df.stc.apply(lambda x: self.embed_model.encode(x).tolist())
        self.df['stc_embed'] = stc_embed
    
    def _search_query_embedding(self, query):
        if self.ds_id == 'arguana':
            query_embedding = self.embed_model.encode(query)
            psg_sim = self.df.psg_embed.apply(lambda x: cosine_similarity(x, query_embedding))
            self.df['stc_sim'] = psg_sim
        else:
            query_embedding = self.embed_model.encode_queries(query)
            stc_scores = self.df.stc_embed.apply(lambda x: [cosine_similarity(xx,query_embedding) for xx in x])
            self.df['stc_scores'] = stc_scores
            stc_sim = self.df['stc_scores'].apply(lambda x: max(x))
            self.df['stc_sim'] = stc_sim
       
        
        return self.df.sort_values('stc_sim', ascending=False, ignore_index=True)
    
    def search_query(self, query, head_num):
        res = self._search_query_embedding(query).head(head_num)
        return res['idx'].tolist(),res['text'].tolist()
    
    def search_queries(self, query_path, output_path, head_num=10):
        qdf = pd.read_json(query_path)
        print('retrieve start')
        t = time.time()
        res_dicts = []
        for i in range(qdf.shape[0]):
            query = qdf.loc[i,'query']
            idxs, texts = self.search_query(query,head_num)
            res_dicts.append({'qid': qdf.loc[i,'qid'], 'query': query, 'text': 'SEP###_###PES'.join(texts), 'refs': qdf.loc[i,'refs'], 'pages': idxs})
        res_df = pd.DataFrame(res_dicts)
        res_df.to_json(output_path,orient='records',lines=True)
        print(time.time()-t)