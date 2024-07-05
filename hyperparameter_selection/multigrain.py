import pandas as pd
import numpy as np
import time
from FlagEmbedding import FlagModel


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class BgeMultigrainSearcher:
    '''
    doc_path: A JSONL file, with `id`, `text`, `stc` as keys for each line.
    name: The dataset's name. For `arguana`, we consider their queries are long enough that they don't need fine-grained information from corpus.
    '''
    def __init__(self, doc_path,name):
        self.df = self._construct_df(doc_path)
        self.ds_id = name
        self.embed_model = FlagModel('bge-large-en-v1.5',query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        self._calculate_psg_embed()
        self._calculate_stc_embed()
    
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

        if 'stc_embed' in df.columns:
            self.has_stc_embed = True
        else:
            self.has_stc_embed = False
        return df
    
    def _calculate_psg_embed(self):
        if not self.has_embed:
            print('calculate embedding')
            psg_embed = self.df.text.apply(lambda x: self.embed_model.encode(x))
            self.df['psg_embed'] = psg_embed
        else:
            print('use embedding')
    
    def _calculate_stc_embed(self):
        if not self.has_stc_embed:
            print('calculate stc embedding')
            stc_embed = self.df.sentences.apply(lambda x: self.embed_model.encode(x).tolist())
            self.df['stc_embed'] = stc_embed
        else:
            print('use stc embedding')
    
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
            return self.df.sort_values("ensemble_sim", ascending=False, ignore_index=True)
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
            
            return self.df.sort_values("ensemble_sim", ascending=False, ignore_index=True)

    def search_query(self, query,a,b,l, head_num):
        res = self._search_mg_embed(query,a,b,l).head(head_num)
        return res['idx'].tolist(),res['text'].tolist()
    
    def search_queries(self, query_path, para_dict, head_num=10):
        qdf = pd.read_json(query_path)
        print('retrieve start')
        alphas = para_dict['alpha']
        betas = para_dict['beta']
        ls = para_dict['l']
        for l in ls:
            for alpha in alphas:
                for beta in betas:
                    t = time.time()
                    output_path = f'para_mg_l_alpha_beta/{self.ds_id}_mulgrain_l{int(l*100)}_a{alpha}b{beta}.jsonl'
                    print(output_path)
                    res_dicts = []
                    for i in range(qdf.shape[0]):
                        query = qdf.loc[i,'query']
                        idxs, texts = self.search_query(query,alpha,beta,l,head_num)
                        res_dicts.append({'qid': qdf.loc[i,'qid'], 'query': query, 'text': 'SEP###_###PES'.join(texts), 'refs': qdf.loc[i,'refs'], 'pages': idxs})
                    res_df = pd.DataFrame(res_dicts)
                    res_df.to_json(output_path,orient='records',lines=True)
                    print('time: %.2f'%(time.time()-t))

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
    
    datanames = ['scifact','nfcorpus','squad'] # 'arguana',
    para_dict = {'alpha':[50, 60, 70, 80, 90],
                 'beta':[10, 20, 30, 40, 50],
                 'l':[0.05, 0.1,0.15,0.2,0.25]}
    for d in datanames:
        corpus_path, query_path, pyse_index = data_select(d)
        searcher = BgeMultigrainSearcher(corpus_path,d)
        
        searcher.search_queries(query_path,para_dict)