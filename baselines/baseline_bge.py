import pandas as pd
import numpy as np
import time
from FlagEmbedding import FlagModel

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class BgeSearcher:
    '''
    doc_path: a JSONL file, with `id`, `text` as keys for each line.
    '''
    def __init__(self, doc_path):
        self.df = self._construct_df(doc_path)
        self.ds_id = doc_path.split('/')[1]
        self.embed_model = FlagModel('bge-large-en-v1.5',query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        self._calculate_psg_embed()
    
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
    
    def _search_query_embedding(self, query):
        if self.ds_id == 'arguana':
            query_embedding = self.embed_model.encode(query)
        else:
            query_embedding = self.embed_model.encode_queries(query)
        psg_sim = self.df.psg_embed.apply(lambda x: cosine_similarity(x, query_embedding))
        self.df['psg_sim'] = psg_sim
        return self.df.sort_values('psg_sim', ascending=False, ignore_index=True)
    
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