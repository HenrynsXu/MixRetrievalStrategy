import pandas as pd
import spacy
import json,os,re
from collections import defaultdict
nlp = spacy.load('en_core_web_sm')

# scifact

def load_scifact():
    # corpus
    corpus = []
    with open('datasets-raw/scifact/corpus.jsonl','r',encoding='utf-8') as finc:
        for line in finc.readlines():
            d = json.loads(line)
            corpus.append({'idx': d['doc_id'], 'text': ' '.join(d['abstract']), 'stc': d['abstract']})
    
    # queries
    queries = []
    with open('datasets-raw/scifact/claims_dev.jsonl','r',encoding='utf-8') as finq:
        for l in finq.readlines():
            q = json.loads(l)
            if q['evidence']:
                queries.append({'qid':q['id'], 'query': q['claim'], 'refs': [int(k) for k in list(q['evidence'].keys())]})

    with open('datasets/scifact/corpus.json','w',encoding='utf-8') as fc:
        json.dump(corpus,fc,indent=4,ensure_ascii=False)
    with open('datasets/scifact/query.json','w',encoding='utf-8') as fq:
        json.dump(queries,fq,indent=4,ensure_ascii=False)

# nfcorpus

def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def direct_split(text, sep = 21):
    txts = text.split(' ')
    res = []
    for i in range(0, len(txts), sep):
        res.append(' '.join(txts[i:min(i+sep,len(txts))]))
    return res

def load_nfcorpus():
    # corpus
    dfc = pd.read_csv('datasets-raw/nfcorpus/test.docs', delimiter='\t',names=['idx','text'])
    corpus = []
    big_cnt = 0
    for i in range(dfc.shape[0]):
        corpus.append({'idx':dfc.loc[i,'idx'],'text':dfc.loc[i,'text'], 'stc':direct_split(dfc.loc[i,'text'])})
        if len(dfc.loc[i,'text'].split(' '))> 500: big_cnt+=1
    print(f'# {big_cnt} in {len(corpus)}')

    # queries and qrl
    dfq = pd.read_csv('datasets-raw/nfcorpus/test.vid-titles.queries', delimiter='\t',names=['qid','query'])
    dfqrel = pd.read_csv('datasets-raw/nfcorpus/test.2-1-0.qrel', delimiter='\t',names=['qid','zero','idx','label'])
    temp_dict = defaultdict(list)
    for i in range(dfqrel.shape[0]):
        if dfqrel.loc[i,'label']==2:
            temp_dict[dfqrel.loc[i,'qid']].append(dfqrel.loc[i,'idx'])
    queries = []
    for i in range(dfq.shape[0]):
        queries.append({'qid':dfq.loc[i,'qid'],'query':dfq.loc[i,'query'], 'refs':temp_dict[dfq.loc[i,'qid']]})

    with open('datasets/nfcorpus/corpus.json','w',encoding='utf-8') as fc:
        json.dump(corpus,fc,indent=4,ensure_ascii=False)
    with open('datasets/nfcorpus/query.json','w',encoding='utf-8') as fq:
        json.dump(queries,fq,indent=4,ensure_ascii=False)

# arguana

def get_directories(path):

    return [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

def extract_prefix_and_text(texts):
    results = []
    pattern = re.compile(r'^(PRO|CON)\d{2}(A|B)-(POINT|COUNTER)?')
    
    for text in texts:
        if not text: continue
        match = pattern.match(text)
        if match:
            prefix = match.group()
            rest_of_text = text[match.end():].strip()
            results.append((prefix.lower(), rest_of_text))
        else:
            results.append((None, text))
    
    return results

def load_arguana():
    # corpus
    title_dict = {}
    corpus = []
    big_cnt = 0
    firsts = get_directories('test')
    for fir in firsts:
        seconds = get_directories(fir)
        for sec in seconds:
            full_text_path = os.path.join(sec,'full.txt')
            with open(full_text_path,'r',encoding='utf-8') as f:
                temp_s = f.read()
            extracts = extract_prefix_and_text([ss.strip() for ss in temp_s.split('#')])
            for ext in extracts:
                if not ext[0]: continue
                if ext[0][:3] == 'pro':
                    title = sec+'/pro/'+ext[0] + '.txt'
                else:
                    title = sec+'/con/'+ext[0] + '.txt'
                title = title.replace('root/retr-copy/datasets-raw/arguana/','')
                corpus.append({'idx':title, 'text': ext[1], 'stc':split_into_sentences(ext[1])})
                title_dict[title] = ext[1]
                if len(ext[1].split(' '))> 500: big_cnt+=1
    print(f'# {big_cnt} in {len(corpus)}')
    # queries
    queries = []
    dfq = pd.read_csv('01-debate-opposing-counters.tsv', delimiter='\t',names=['qtitle','candi','s1','s2','s3','s4'])
    for i in range(dfq.shape[0]):
        if dfq.loc[i,'s1']:
            queries.append({'qid':dfq.loc[i,'qtitle'],'query':title_dict[dfq.loc[i,'qtitle']],'refs':dfq.loc[i,'candi']})

    with open('root/retr-copy/datasets/arguana/corpus.json','w',encoding='utf-8') as fc:
        json.dump(corpus,fc,indent=4,ensure_ascii=False)
    with open('root/retr-copy/datasets/arguana/query.json','w',encoding='utf-8') as fq:
        json.dump(queries,fq,indent=4,ensure_ascii=False)

# squad

import random
random.seed(2024)
def load_squad():
    corpus = []
    queries = []
    with open('datasets-raw/squad-dev-v1.1.json','r',encoding='utf-8') as f:
        data = json.load(f)['data']
        for i, doc in enumerate(data):
            pars = doc['paragraphs']
            for j,par in enumerate(pars):
                text = par['context']
                qas = par['qas']
                passage_id = 'd%02dp%02d'%(i,j)
                corpus.append({'idx':passage_id, 'text':text, 'stc': split_into_sentences(text)})
                random.shuffle(qas)
                sample_qa = qas[0]
                queries.append({'qid': sample_qa['id'], 'query': sample_qa['question'], 'refs': passage_id})
    
    with open('datasets/squad/corpus.json','w',encoding='utf-8') as fc:
        json.dump(corpus,fc,indent=4,ensure_ascii=False)
    with open('datasets/squad/query_all.json','w',encoding='utf-8') as fq:
        json.dump(queries,fq,indent=4,ensure_ascii=False)

if __name__ == '__main__':
    load_squad()