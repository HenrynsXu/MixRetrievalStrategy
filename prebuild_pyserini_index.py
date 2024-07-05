'''
Use the transfer function to get the format required by Pyserini. 
The use the command below (for example) to build indexes.

`python -m pyserini.index.lucene --collection JsonCollection --input pyse_preprocess/squad --index pyse_index/squad --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw --storeContents`

For more information, refer to [Pyserini](https://github.com/castorini/pyserini)[1].

[1] Lin J, Ma X, Lin S C, et al. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021: 2356-2362.
'''


import json

def transfer_to_pyse(dataset):
    if dataset == 'arguana':  # 2807-140
        inpath = 'datasets/arguana/corpus.json'
        outpath = 'pyse_preprocess/arguana.json'
    elif dataset == 'nfcorpus':  # 3462-102
        inpath = 'datasets/nfcorpus/corpus.json'
        outpath = 'pyse_preprocess/nfcorpus.json'
    elif dataset == 'scifact':  # 5183-188
        inpath = 'datasets/scifact/corpus.json'
        outpath = 'pyse_preprocess/scifact.json'
    elif dataset == 'squad': # 2067-110
        inpath = 'datasets/squad/corpus.json'
        outpath = 'pyse_preprocess/squad.json'
    with open(inpath,'r',encoding='utf-8') as f:
        data = json.load(f)
    pyseout = []
    for d in data:
        pyseout.append({'id':str(d['idx']), 'contents': d['text']})
    with open(outpath,'w',encoding='utf-8') as fout:
        json.dump(pyseout,fout,ensure_ascii=False,indent=4)


if __name__ == '__main__':

    transfer_to_pyse('squad')