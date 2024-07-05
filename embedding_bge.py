'''
Get Bge embeddings for our corpus
'''

import json
from tqdm import tqdm
from FlagEmbedding import FlagModel

bge_model = FlagModel('BAAI/bge-large-en-v1.5',query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')

def get_bge_embedding(text,):  # only a string
    result = bge_model.encode(text).tolist()
    return result

def add_embedding_to_corpus(name):
    if name == 'arguana':  # 2807-140
        inpath = 'datasets/arguana/corpus.json'
        outpath = 'datasets-bge/arguana_corpus.json'
    elif name == 'nfcorpus':  # 3462-102
        inpath = 'datasets/nfcorpus/corpus.json'
        outpath = 'datasets-bge/nfcorpus_corpus.json'
    elif name == 'scifact':  # 5183-188
        inpath = 'datasets/scifact/corpus.json'
        outpath = 'datasets-bge/scifact_corpus.json'
    elif name == 'squad': # 2067-110
        inpath = 'datasets/squad/corpus.json'
        outpath = 'datasets-bge/squad_corpus.json'

    output_with_embed = []
    with open(inpath,'r',encoding='utf-8') as fin:
        datas = json.load(fin)
    for data in tqdm(datas,desc=f'embedding bge for passage in {name}'):
        psg_embed = get_bge_embedding(data['text'])
        stc_embed = [get_bge_embedding(stc) for stc in data['stc']]
        
        output_with_embed.append({'idx':data['idx'], 'text': data['text'], 'stc': data['stc'], 'psg_embed': psg_embed, 'stc_embed': stc_embed})
    with open(outpath,'w',encoding='utf-8') as fout:
        json.dump(output_with_embed, fout, indent=4, ensure_ascii=False)

if __name__ =='__main__':
    names = ['arguana','nfcorpus','scifact','squad']
    for name in names:
        add_embedding_to_corpus(name)
        