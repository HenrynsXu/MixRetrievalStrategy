'''
Get OpenAI embeddings for our corpus
'''

import json
import os
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key='')

def get_openai_embedding(text,max_retry=100):  # only a string
    for retry_time in range(max_retry):
        try:
            result = client.embeddings.create(input=text,model='text-embedding-3-large').data[0].embedding
            return result
        except Exception as e:
            print(f"Error on attempt {retry_time + 1}: {e}")
            if retry_time < max_retry - 1:
                print("Retrying...")
    print(f"Max attempts ({max_retry}) reached. Unable to call OpenAI Embedding.")
    return None

def add_embedding_to_corpus(name):
    outdir = 'datasets-openai3'
    if name == 'arguana':  # 2807-140
        inpath = 'datasets/arguana/corpus.json'
        outpath = os.path.join(outdir,'arguana_corpus.json')
    elif name == 'nfcorpus':  # 3462-102
        inpath = 'datasets/nfcorpus/corpus.json'
        outpath = os.path.join(outdir,'nfcorpus_corpus.json')
    elif name == 'scifact':  # 5183-188
        inpath = 'datasets/scifact/corpus.json'
        outpath = os.path.join(outdir,'scifact_corpus.json')
    elif name == 'squad': # 2067-110
        inpath = 'datasets/squad/corpus.json'
        outpath = os.path.join(outdir,'squad_corpus.json')

    output_with_embed = []
    with open(inpath,'r',encoding='utf-8') as fin:
        datas = json.load(fin)
    for data in tqdm(datas,desc=f'embedding text 3 for passage in {name}'):
        psg_embed = get_openai_embedding(data['text'])
        stc_embed = [get_openai_embedding(stc) for stc in data['stc']]
        
        output_with_embed.append({'idx':data['idx'], 'text': data['text'], 'stc': data['stc'], 'psg_embed': psg_embed, 'stc_embed': stc_embed})
    with open(outpath,'w',encoding='utf-8') as fout:
        json.dump(output_with_embed, fout, indent=4, ensure_ascii=False)

def add_embedding_to_query(name):
    outdir = 'datasets-openai3'
    if name == 'arguana':  # 2807-140
        inpath = 'datasets/arguana/query_test.json'
        outpath = os.path.join(outdir,'arguana_query_test.json')
    elif name == 'nfcorpus':  # 3462-102
        inpath = 'datasets/nfcorpus/query.json'
        outpath = os.path.join(outdir,'nfcorpus_query.json')
    elif name == 'scifact':  # 5183-188
        inpath = 'datasets/scifact/query.json'
        outpath = os.path.join(outdir,'scifact_query.json')
    elif name == 'squad': # 2067-110
        inpath = 'datasets/squad/query_test.json'
        outpath = os.path.join(outdir,'squad_query_test.json')

    output_with_embed = []
    with open(inpath,'r',encoding='utf-8') as fin:
        datas = json.load(fin)
    for data in tqdm(datas,desc=f'embedding openai text 3 for query in {name}'):
        q_embed = get_openai_embedding(data['query'])
        
        output_with_embed.append({'qid':data['qid'], 'query': data['query'], 'refs': data['refs'], 'q_embed': q_embed})
    with open(outpath,'w',encoding='utf-8') as fout:
        json.dump(output_with_embed, fout, indent=4, ensure_ascii=False)

if __name__ =='__main__':
    names = ['arguana','nfcorpus','scifact','squad']
    for name in names:
        add_embedding_to_corpus(name)
        add_embedding_to_query(name)