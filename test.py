THE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'arguana': 'beir-v1.0.0-arguana.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat',
    'scidocs': 'beir-v1.0.0-scidocs.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'quora': 'beir-v1.0.0-quora.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'fever': 'beir-v1.0.0-fever-flat',
    'robust04': 'beir-v1.0.0-robust04.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',

    'mrtydi-ar': 'mrtydi-v1.1-arabic',
    'mrtydi-bn': 'mrtydi-v1.1-bengali',
    'mrtydi-fi': 'mrtydi-v1.1-finnish',
    'mrtydi-id': 'mrtydi-v1.1-indonesian',
    'mrtydi-ja': 'mrtydi-v1.1-japanese',
    'mrtydi-ko': 'mrtydi-v1.1-korean',
    'mrtydi-ru': 'mrtydi-v1.1-russian',
    'mrtydi-sw': 'mrtydi-v1.1-swahili',
    'mrtydi-te': 'mrtydi-v1.1-telugu',
    'mrtydi-th': 'mrtydi-v1.1-thai',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}

from rank_gpt import run_retriever, sliding_windows, write_eval_file
from pyserini.search import LuceneSearcher, get_topics, get_qrels
from tqdm import tqdm
import tempfile
import os
import json
import shutil
import argparse
import json
import configparser

config = configparser.ConfigParser()
config.read('../.config')
openai_key = config.get('DEFAULT', 'OPENAI_API_KEY')
# dl19 68.84
# dl20 63.02
# covid 75.91
# nfc 36.52
# touche 35.60
# dbpedia 43.29
# scifact 

parser = argparse.ArgumentParser(description='LLM Reranker')
parser.add_argument('--prompt_type', type=int, default=8)
parser.add_argument('--dataset', type=str, default='dl19')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
parser.add_argument('--print_messages', type=int, default=0)
parser.add_argument('--correct_malform', type=int, default=1)
parser.add_argument('--debug', type=int, default=1)
args = parser.parse_args()
print(json.dumps(vars(args), indent=4))

args.rep_query_cnt = 0
args.rep_passage_cnt = 0
args.miss_query_cnt = 0
args.miss_passage_cnt = 0
args.mismatch_query_cnt = 0


for data in [args.dataset]:
    print('#' * 20)
    print(f'Evaluation on {data}')
    print('#' * 20)

    
    # Retrieve passages using pyserini BM25.
    try:
        searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
        topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
        qrels = get_qrels(THE_TOPICS[data])
        rank_results = run_retriever(topics, searcher, qrels, k=100)
    except:
        print(f'Failed to retrieve passages for {data}')
        continue
    
    # Run sliding window permutation generation
    new_results = []
    for i, item in enumerate(rank_results):
        print('*'*100)
        print(f'query {i}/{len(rank_results)}: ')
        new_item = sliding_windows(args, item, rank_start=0, rank_end=100, window_size=20, step=10,
                                   model_name=args.model, api_key=openai_key)
        new_results.append(new_item)
        print('*'*100)
        if args.debug:
            break
    print(
        'Total Malformed Outputs Statistics:\n',
       f'Total Repetition Chat Count: {args.rep_query_cnt}, Average Repetition Query: {args.rep_query_cnt / (len(rank_results) * 9)}\n',
       f'Total Repetition Passage Count: {args.rep_passage_cnt}, Average Repetition Passage: {args.rep_passage_cnt / (len(rank_results)*9*20)}\n',
       f'Total Missing Chat Count: {args.miss_query_cnt}, Average Missing Query: {args.miss_query_cnt / (len(rank_results)*9)}\n',
       f'Total Missing Passage Count: {args.miss_passage_cnt}, Average Missing Passage: {args.miss_passage_cnt / (len(rank_results)*9*20)}\n',
       f'Total Mismatch Chat Count: {args.mismatch_query_cnt}, Average Mismatch Query: {args.mismatch_query_cnt / (len(rank_results)*9)}\n')
    # Evaluate nDCG@10
    from trec_eval import EvalFunction

    # Create an empty text file to write results, and pass the name to eval
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(new_results, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], output_file])
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.5', THE_TOPICS[data], output_file])
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.1', THE_TOPICS[data], output_file])
    # Rename the output file to a better name
    shutil.move(output_file, f'eval_{data}.txt')


# for data in ['mrtydi-ar', 'mrtydi-bn', 'mrtydi-fi', 'mrtydi-id', 'mrtydi-ja', 'mrtydi-ko', 'mrtydi-ru', 'mrtydi-sw', 'mrtydi-te', 'mrtydi-th']:
#     print('#' * 20)
#     print(f'Evaluation on {data}')
#     print('#' * 20)

#     # Retrieve passages using pyserini BM25.
#     try:
#         searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
#         topics = get_topics(THE_TOPICS[data] if data != 'dl20' else 'dl20')
#         qrels = get_qrels(THE_TOPICS[data])
#         rank_results = run_retriever(topics, searcher, qrels, k=100)
#         rank_results = rank_results[:100]

#     except:
#         print(f'Failed to retrieve passages for {data}')
#         continue

#     # Run sliding window permutation generation
#     new_results = []
#     for item in tqdm(rank_results):
#         new_item = sliding_windows(args, item, rank_start=0, rank_end=100, window_size=20, step=10,
#                                    model_name='gpt-3.5-turbo', api_key=openai_key)
#         new_results.append(new_item)

#     # Evaluate nDCG@10
#     from trec_eval import EvalFunction

#     temp_file = tempfile.NamedTemporaryFile(delete=False).name
#     write_eval_file(new_results, temp_file)
#     EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', THE_TOPICS[data], temp_file])
#         # Rename the output file to a better name
#     shutil.move(output_file, f'eval_{data}.txt')