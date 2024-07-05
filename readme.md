## Code For Our Retrieval Strategy for RAG
This is the code reposity for our mixed retrieval strategy, including sparse-dense hybrid and multi-granularity retrieval. 

### Quick Usage

You can quickly call our `mixretriever_quickuse.py` file, which currently supports both Chinese and English. Simply provide the query, the IDs of the candidate paragraphs, the first-stage retrieval scores, and the IDF values of each token in the corresponding corpus.

```python
from mixretriever_quickuse import quick_search
quick_search('your query', 'id_scores.jsonl', idf_dictionary, lang='en')
```

After running, you can view the final paragraph ID rankings and the fused scores in `output.jsonl`.

### Important Files Description
- For our experiments, the input corpus are JSONL files, whose each line contains a passage represented by JSON string, including `id` (the passage id), `text` (passage's text), `stc` (sentences splited by the passage text).
- The `baselines` folder contains the baseline experiments we used, including the latest bge-m3 model.
- The `hyperparameter_selection` folder contains the code for our hyperparameter tuning experiments.
- `data_preprocess.py` preprocesses the raw dataset (extracted from zip files) into the desired format, making it easy to divide into validation and test sets.
- `embedding_[bge,openai].py` encodes the corpus (and queries) and stores them locally for later use.
- `mixretriever_[bge,openai].py` contains the final implementation of our solution, with different files representing the use of different encoding models. Whether to use bag-of-words BM25 or vanilla BM25 depends on whether the path to the bag-of-words index is provided during initialization.
- `prebuild_pyserini_index.py` is used to process the dataset into the format required to generate BM25 indexes using the Pyserini library. For more information, refer to [Pyserini](https://github.com/castorini/pyserini)[1].
- `retrieve_launch.py` can launch and implement our solution as well as the baselines.

[1] Lin J, Ma X, Lin S C, et al. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021: 2356-2362.