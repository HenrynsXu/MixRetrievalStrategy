from baselines.baseline_bm25 import Bm25BasicSearcher
from baselines.baseline_bge import BgeSearcher
from baselines.baseline_bge_stc import BgeStcSearcher
from baselines.baseline_bgem3 import Bgem3Searcher
from baselines.baseline_pyserini_bm25 import Bm25PyseriniSearcher
from baselines.baseline_openai import OpenaiSearcher
from baselines.baseline_naive_hybrid import NaiveHybridSearcher
from baselines.baseline_naive_multigrained import BgeNaiveMultigrainSearcher
from mixretriever_bge import MixPyseriniSearcher
from mixretriever_openai import MixPyseriniOpenaiSearcher

def data_select(name):
    if name == 'scifact':
        return ('datasets/scifact/corpus.json', 'datasets/scifact/query_test.json', 'pyse_index/scifact')
    elif name == 'nfcorpus':
        return ('datasets/nfcorpus/corpus.json', 'datasets/nfcorpus/query_test.json', 'pyse_index/nfcorpus')
    elif name == 'arguana':
        return ('datasets/arguana/corpus.json', 'datasets/arguana/query_test.json', 'pyse_index/arguana')
    elif name == 'squad':
        return ('datasets/squad/corpus.json', 'datasets/squad/query_test.json', 'pyse_index/squad')
    else: raise

def data_select_openai(name):
    if name == 'scifact':
        return ('/data/datasets-openai/scifact_corpus.json', '/data/datasets-openai/scifact_query_test.json', 'pyse_index/scifact')
    elif name == 'nfcorpus':
        return ('/data/datasets-openai/nfcorpus_corpus.json', '/data/datasets-openai/nfcorpus_query_test.json', 'pyse_index/nfcorpus')
    elif name == 'arguana':
        return ('/data/datasets-openai/arguana_corpus.json', '/data/datasets-openai/arguana_query_test.json', 'pyse_index/arguana')
    elif name == 'squad':
        return ('/data/datasets-openai/squad_corpus.json', '/data/datasets-openai/squad_query_test.json', 'pyse_index/squad')
    else: raise

def data_select_bge(name):
    if name == 'scifact':
        return ('datasets-bge/scifact_corpus.json', 'datasets/scifact/query_test.json', 'pyse_index/scifact')
    elif name == 'nfcorpus':
        return ('datasets-bge/nfcorpus_corpus.json', 'datasets/nfcorpus/query_test.json', 'pyse_index/nfcorpus')
    elif name == 'arguana':
        return ('datasets-bge/arguana_corpus.json', 'datasets/arguana/query_test.json', 'pyse_index/arguana')
    elif name == 'squad':
        return ('datasets-bge/squad_corpus.json', 'datasets/squad/query_test.json', 'pyse_index/squad')
    else: raise
    
datanames = ['scifact','nfcorpus','arguana','squad'] # 
for d in datanames:
    corpus_path, query_path, pyse_index = data_select_bge(d)
    searcher = MixPyseriniSearcher(corpus_path,d )  # pyse_index
    outpath = f'log_{d}/final.jsonl'
    print(outpath)
    searcher.search_queries(query_path,outpath)

