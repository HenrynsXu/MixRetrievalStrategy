import pandas as pd
import codecs

import time
from ..utils.vanilla_bm25 import BM25
import spacy


class Bm25BasicSearcher:
    '''
    doc_path: a JSONL file, with `id`, `text` as keys for each line.
    '''
    def __init__(self, doc_path):
        self.df = self._construct_df(doc_path)
        self.ds_id = doc_path.split('/')[1]
        self.en_lang = spacy.load('en_core_web_sm')
        self._build_basic_bm25()
    
    def _construct_df(self, doc_path):
        '''
        format: a list of json dicts
        '''
        df = pd.read_json(doc_path)
        assert 'text' in df.columns
        return df
    
    def _build_basic_bm25(self):
        
        stopwords = codecs.open('utils/stopwords.txt','r',encoding='utf8').readlines()
        self.stop_flag = ['x', 'c', 'u','d', 'p', 't', 'uj', 'm', 'f', 'r'] 
        self.en_stop_flag = {"DET", "PRON", "CONJ", "SCONJ", "PUNCT", "NUM", "SYM", "X", "SPACE"}
        self.stopwords = [ w.strip() for w in stopwords ]
        self.df['wordbag']=self.df.text.apply(lambda x:self._tokenization(x))
        word_list=self.df.wordbag.tolist()
        self.bm25=BM25(word_list)

    def _tokenization(self,text):
        doc = self.en_lang(text)
        return [token.text for token in doc if token.text not in self.stopwords] # token.pos_ not in self.en_stop_flag and 
    
    def _search_basic_bm25(self, query):
        query_tokens = []
        for token in self.en_lang(query):
            query_tokens.append(token.text)
        scores = self.bm25.get_scores(query_tokens)
        self.df['bm25_sim']=scores
        return self.df.sort_values('bm25_sim', ascending=False, ignore_index=True)
    
    def search_query(self, query, head_num):
        res = self._search_basic_bm25(query).head(head_num)
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
