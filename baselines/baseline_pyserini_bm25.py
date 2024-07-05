import pandas as pd
import codecs
import json
import time
from pyserini.search.lucene import LuceneSearcher
import spacy

class Bm25PyseriniSearcher:
    '''
    doc_path: a JSONL file, with `id`, `text` as keys for each line.
    '''
    def __init__(self, doc_path, pyse_index):
        self.df = self._construct_df(doc_path)
        self.ds_id = doc_path.split('/')[1]
        self.en_lang = spacy.load('en_core_web_sm')
        self._build_pyse_bm25(pyse_index)
    
    def _construct_df(self, doc_path):
        '''
        format: a list of json dicts
        '''
        df = pd.read_json(doc_path)
        assert 'text' in df.columns
        return df
    
    def _build_pyse_bm25(self,index):
        self.pysearcher = LuceneSearcher(index)
    
    def _search_pyse_bm25(self, query):
        retrs = self.pysearcher.search(query,self.df.shape[0])
        pyse_dic = {retrs[i].docid: retrs[i].score for i in range(len(retrs))}
        bm25_sim = self.df['idx'].apply(lambda x: pyse_dic.get(str(x),0))
        self.df['bm25_sim'] = bm25_sim
        return self.df.sort_values('bm25_sim', ascending=False, ignore_index=True)
    
    def search_query(self, query, head_num):
        res = self._search_pyse_bm25(query).head(head_num)
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
