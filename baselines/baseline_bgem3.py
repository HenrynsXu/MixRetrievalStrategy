from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import numpy as np
import time


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Bgem3Searcher:
    '''
    doc_path: a JSONL file, with `id`, `text` as keys for each line.
    '''
    def __init__(self, doc_path):
        self.df = self._construct_df(doc_path)
        self.ds_id = doc_path.split('/')[1]
        self.m3_model = BGEM3FlagModel('bge-m3')
        self._calculate_embed()
    
    def _construct_df(self, doc_path):
        '''
        format: a list of json dicts
        '''
        df = pd.read_json(doc_path)
        assert 'text' in df.columns
        return df
    
    def _calculate_embed(self):
        text = self.df.text.tolist()
        output = self.m3_model.encode(text,return_dense=True,return_colbert_vecs=True,return_sparse=True)
        dense = output['dense_vecs'].tolist()
        sparse = output['lexical_weights']
        colbert = output['colbert_vecs']
        
        self.df['dense'] = dense
        self.df['sparse'] = sparse
        self.df['colbert'] = colbert
    
    def _search_query_embedding(self, query):
        weight = [1., 1., 1.]
        query_output = self.m3_model.encode(query,return_colbert_vecs=True,return_sparse=True)

        dense_sim = self.df.dense.apply(lambda x: cosine_similarity(x, query_output['dense_vecs']))
        self.df['dense_sim'] = dense_sim
        sparse_sim = self.df['sparse'].apply(lambda x: self.m3_model.compute_lexical_matching_score(x, query_output['lexical_weights']))
        self.df['sparse_sim'] = sparse_sim
        colbert_sim = self.df.colbert.apply(lambda x: self.m3_model.colbert_score(x,query_output['colbert_vecs']))
        self.df['colbert_sim'] = colbert_sim
        avg_sim = np.average([self.df.dense_sim.tolist(), self.df.sparse_sim.tolist(), self.df.colbert_sim.tolist()],axis=0,weights=weight)
        self.df['avg_sim'] = avg_sim
        return self.df.sort_values('avg_sim', ascending=False, ignore_index=True)
    
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