import pandas as pd
import numpy as np
import codecs
import json
import time
from ..utils.vanilla_bm25 import BM25
from FlagEmbedding import FlagModel
import spacy

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalize(sery):
    mins,maxs = min(sery),max(sery)
    for i in range(len(sery)):
        sery[i] = (sery[i]-mins)/(maxs-mins+1e-6)
    return sery

class MixSearcher:
    '''
    doc_path: A JSONL file, with `id`, `text`, `stc` as keys for each line.
    name: The dataset's name. For `arguana`, we consider their queries are long enough that they don't need fine-grained information from corpus.
    '''
    def __init__(self, doc_path,name):
        self.df = self._construct_df(doc_path)
        self.ds_id = name
        self.en_lang = spacy.load('en_core_web_sm')
        self._build_basic_bm25()
        self.embed_model = FlagModel('bge-large-en-v1.5',query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        self._calculate_psg_embed()
    
    def _construct_df(self, doc_path):
        '''
        format: a list of json dicts
        '''
        df = pd.read_json(doc_path)
        assert 'text' in df.columns

        if 'psg_embed' in df.columns:
            self.has_embed = True
        else:
            self.has_embed = False
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
        return [token.text for token in doc if token.text not in self.stopwords]  # token.pos_ not in self.en_stop_flag and 
    
    def _calculate_psg_embed(self):
        if not self.has_embed:
            print('calculate embedding')
            psg_embed = self.df.text.apply(lambda x: self.embed_model.encode(x))
            self.df['psg_embed'] = psg_embed
        else:
            print('use embedding')

    def _search_basic_bm25(self, query):
        query_tokens = []
        for token in self.en_lang(query):
            query_tokens.append(token.text)
        scores = self.bm25.get_scores(query_tokens)
        self.df['bm25_sim']=scores
        # return self.df.sort_values('bm25_sim', ascending=False, ignore_index=True)
    
    def _search_psg_embedding(self, query):
        if self.ds_id == 'arguana':
            query_embedding = self.embed_model.encode(query)
        else:
            query_embedding = self.embed_model.encode_queries(query)
        psg_sim = self.df.psg_embed.apply(lambda x: cosine_similarity(x, query_embedding))
        self.df['psg_sim'] = psg_sim
        return self.df.sort_values('psg_sim', ascending=False, ignore_index=True)

    def _search_stc_embedding(self, query):
        if self.ds_id == 'arguana':
            query_embedding = self.embed_model.encode(query)
            psg_sim = self.df.psg_embed.apply(lambda x: cosine_similarity(x, query_embedding))
            self.df['stc_sim'] = psg_sim
        else:
            # query_embedding = self.embed_model.encode_queries(query)
            
            stc_sim = self.df['stc_scores'].apply(lambda x: max(x))
            self.df['stc_sim'] = stc_sim
        return self.df.sort_values('stc_sim', ascending=False, ignore_index=True)
    
    def _search_mg_embed(self, query, alpha,beta,l):
        if self.ds_id == 'arguana':
            dummy = self._search_psg_embedding(query)
            self.df['ensemble_sim'] = self.df['psg_sim']
            # return self.df.sort_values("ensemble_sim", ascending=False, ignore_index=True)
        else:
            query_embedding = self.embed_model.encode_queries(query)
            stc_scores = self.df.stc_embed.apply(lambda x: [cosine_similarity(xx,query_embedding) for xx in x])
            self.df['stc_scores'] = stc_scores
            self.df['range'] = self.df['stc_scores'].apply(lambda x: max(x)-min(x))
            psg_res = self._search_psg_embedding(query).head(5)
            stc_res = self._search_stc_embedding(query).head(5)
            p_range = np.mean(psg_res['range'].tolist())
            s_range = np.mean(stc_res['range'].tolist())
            if s_range-p_range<1e-6 or s_range-p_range>l:
                weights = [100-alpha,alpha]
            else:
                weights = [100-beta,beta]
            
            self.df['ensemble_sim'] = np.average([self.df.psg_sim.tolist(),
                                                  self.df.stc_sim.tolist()],
                                                  weights=weights,
                                                  axis=0)
            
            # return self.df.sort_values("ensemble_sim", ascending=False, ignore_index=True)

    def _search_hybrid(self, query: str, thres, weight_s, ):
        query_tokens = []
        for token in self.en_lang(query):
            query_tokens.append(token.text)
        idf_dict = self.bm25.idf
        query_idfs = [idf_dict.get(q,0) for q in query]
        query_idfs.sort(reverse=True)
        values = list(idf_dict.values())
        values.sort(reverse=True)
        top1_thres = values[int((thres-0.1)*len(values))]
        topk_thres = values[int(thres*len(values))]
        if query_idfs[0]>top1_thres and np.mean(query_idfs[:3]) > topk_thres:
            self.df['ensemble_c1n'] = np.average([normalize(self.df.bm25_sim.tolist()),
                                                  normalize(self.df.ensemble_sim.tolist())],
                                                 axis=0,
                                                 weights=[weight_s,1-weight_s])
        else:
            self.df['ensemble_c1n'] = np.average([normalize(self.df.bm25_sim.tolist()),
                                                  normalize(self.df.ensemble_sim.tolist())],
                                                  axis=0,
                                                  weights=[1-weight_s,weight_s])
        return self.df.sort_values("ensemble_c1n", ascending=False, ignore_index=True)

    def search_query(self, query, weight, idfthreshold, head_num):
        self._search_basic_bm25(query)
        self._search_mg_embed(query,alpha=50,beta=40,l=0.1)

        res = self._search_hybrid(query,idfthreshold, weight, ).head(head_num)
        return res['idx'].tolist(),res['text'].tolist()
    
    def search_queries(self, query_path, para_dict,head_num=10):
        qdf = pd.read_json(query_path)
        # if 'q_embed' in qdf.columns:
        #     has_qembed = True
        print('retrieve start')

        weights = para_dict['w']
        idfs = para_dict['idf']
        for w in weights:
            for idf in idfs:
                output_path = f'para_final_mix/{self.ds_id}_hybrid_w{int(w*100)}_idf{int(idf*100)}.jsonl'
                print(output_path)
                res_dicts = []
                for i in range(qdf.shape[0]):
                    query = qdf.loc[i,'query']
                    idxs, texts = self.search_query(query,w, idf,head_num)
                    res_dicts.append({'qid': qdf.loc[i,'qid'], 'query': query, 'text': 'SEP###_###PES'.join(texts), 'refs': qdf.loc[i,'refs'], 'pages': idxs})
                res_df = pd.DataFrame(res_dicts)
                res_df.to_json(output_path,orient='records',lines=True)

if __name__ == '__main__':
    def data_select(name):
        if name == 'scifact':
            return ('../datasets-bge/scifact_corpus.json', '../datasets/scifact/query_dev.json', '../pyse_index/scifact')
        elif name == 'nfcorpus':
            return ('../datasets-bge/nfcorpus_corpus.json', '../datasets/nfcorpus/query_dev.json', '../pyse_index/nfcorpus')
        elif name == 'arguana':
            return ('../datasets-bge/arguana_corpus.json', '../datasets/arguana/query_dev.json', '../pyse_index/arguana')
        elif name == 'squad':
            return ('../datasets-bge/squad_corpus.json', '../datasets/squad/query_dev.json', '../pyse_index/squad')
        else: raise
    
    datanames = ['scifact','nfcorpus','arguana','squad'] # 
    para_dict = {'w':[0.6,0.65,0.7,0.75,0.8],
                 'idf':[0.6, 0.7 , 0.8]}
    for d in datanames:
        corpus_path, query_path, pyse_index = data_select(d)
        searcher = MixSearcher(corpus_path,d)
        
        searcher.search_queries(query_path,para_dict)