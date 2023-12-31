import copy
from tqdm import tqdm
import time
import openai
from openai import AsyncOpenAI
import json
import tiktoken
import re
try:
    from litellm import completion
except:
    completion = openai.ChatCompletion.create

chat_cnt = 0

class SafeOpenai:
    def __init__(self, keys=None, start_id=None, proxy=None):
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        openai.proxy = proxy
        openai.api_key = self.key[self.key_id % len(self.key)]
        self.api_key = self.key[self.key_id % len(self.key)]

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                model = args[0] if len(args) > 0 else kwargs["model"]
                if "gpt" in model:
                    # client = AsyncOpenAI()
                    # async def my_function():
                    #     # Your existing code
                    #     completion = await client.chat.completions.create(*args, **kwargs, timeout=30)
                    #     # Rest of your code
                    # await my_function()
                    completion = openai.chat.completions.create(*args, **kwargs, timeout=60)
                    # completion = openai.ChatCompletion.create(*args, **kwargs, timeout=30)
                elif model in litellm.model_list:
                    completion = completion(*args, **kwargs, api_key=self.api_key, force_timeout=60)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = openai.Completion.create(*args, **kwargs)
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                self.key_id = (self.key_id + 1) % len(self.key)
                openai.api_key = self.key[self.key_id]
                time.sleep(0.1)
        if return_text:
            completion = completion['choices'][0]['text']
        return completion


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message, tokens_per_name = 0, 0

    try:
        encoding = tiktoken.get_encoding(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def max_tokens(model):
    if 'gpt-4' in model:
        return 8192
    else:
        return 4096


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def get_post_cot_prompt(query, num):
    if prompt_type == 1:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. Please use [start] to begin the rank and use [end] to indicate the end of it. The rank format should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. Let's think step by step."
    elif prompt_type == 2:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then give the list of sorted identifiers based on their relevance to the search query. All the passages should be included and listed using identifiers and make sure there is no repetition, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]."
    elif prompt_type == 3:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Please think step by step to solve this task."
    elif prompt_type == 4:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then give the list of sorted identifiers based on their relevance to the search query. All the passages should be included and listed using identifiers and make sure there is no repetition, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]."
    elif prompt_type == 5:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. Use [start] when you begin output the rank and use [end] to indicate the end of the rank. The output format of rank should be [] > [], e.g., [1] > [2]. Please think step by step."
    elif prompt_type == 6:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then give the list of sorted identifiers based on their relevance to the search query. All the passages should be included and listed using identifiers and make sure there is no repetition, in descending order of relevance. At the end of you thought, please summarize the rank and use [start] to begin output the rank and use [end] to indicate the end of the rank. The format of the rank should be [] > [], e.g., [1] > [2]."
    elif prompt_type == 7:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then sort identifiers based on their relevance to the search query. Finally, please summarize the rank and use [start] to begin output the rank and use [end] to indicate the end of the rank. The format of the rank should be [] > [], e.g., [1] > [2]."
    elif prompt_type == 8:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then sort identifiers in descending order based on their relevance to the search query. Finally, summarize the rank. Please use [start] to begin summarize the rank and use [end] to indicate the end of the rank. The format of the summarized rank should be [Start] [] > [] [End], e.g.,[Start] [1] > [2] [End]. In the summarized rank, only response the ranking results, do not say any other explain."
    elif prompt_type == 9:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then sort identifiers in descending order based on their relevance to the search query. Finally, summarize the rank. Please use [start] to begin summarize the rank and use [end] to indicate the end of the rank. All the passages should be included and listed using identifiers and make sure there is no repetition. The format of the summarized rank should be [Start] [] > [] [End], e.g.,[Start] [1] > [2] [End]. In the summarized rank, only response the ranking results, do not say any other explain."
    elif prompt_type == 10:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10. Then sort identifiers in descending order based on their relevance to the search query. Finally, summarize the rank. Please use [start] to begin summarize the rank and use [end] to indicate the end of the rank. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing pasages. The format of the summarized rank should be [Start] [] > [] [End], e.g.,[Start] [1] > [2] [End]. In the summarized rank, only response the ranking results, do not say any other explain."
    elif prompt_type == 11:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10 and explain the reason of rating the score. Then sort identifiers in descending order based on their relevance to the search query. Finally, summarize the rank. Please use [start] to begin summarize the rank and use [end] to indicate the end of the rank. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing pasages. The format of the summarized rank should be [Start] [] > [] [End], e.g.,[Start] [1] > [2] [End]. In the summarized rank, only response the ranking results, do not say any other words."
    elif prompt_type == 12:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 10 and explain the reason of rating the score. Then sort identifiers in descending order based on their relevance to the search query. Finally, summarize the rank. Please use [start] to begin summarize the rank and use [end] to indicate the end of the rank. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing pasages. The format of the summarized rank should be [Start] [] > [] [End], e.g.,[Start] [1] > [2] [End]. In the summarized rank, only response the ranking results."
    elif prompt_type == 13:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step to solve this task. First please rate the relevance between the query and each document from score 0 to 100 and explain the reason of rating the score. Then sort identifiers in descending order based on their relevance to the search query. Finally, summarize the rank. Please use [start] to begin summarize the rank and use [end] to indicate the end of the rank. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing pasages. The format of the summarized rank should be [Start] [] > [] [End], e.g.,[Start] [1] > [2] [End]. In the summarized rank, only response the ranking results."
    elif prompt_type == 14:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step .... First please explain the reasons why you think the query and each passage are relevant or not. Then rate the relevance between the query and each passage from score 0 to 10. After that, please sort identifiers in descending order based on their relevance to the search query. Finally, please summarize the rank and use [start] to begin the summarization and use [end] to indicate the end of it. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing passages. The format of the summarization should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. In the summarization, only response the ranking results."
    elif prompt_type == 15:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step .... First please explain the reasons why you think the query and each passage are relevant or not. Then rate the relevance between the query and each passage from score 0 to 19 with all passages have different scores. After that, please sort identifiers in descending order based on their relevance to the search query. Finally, please summarize the rank and use [start] to begin the summarization and use [end] to indicate the end of it. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing passages. The format of the summarization should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. In the summarization, only response the ranking results."
    elif prompt_type == 16:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step .... First please explain the reasons why you think the query and each passage are relevant or not. Then rate the relevance between the query and each passage from score 0 to 19 with all passages have distinctive scores. After that, please sort identifiers in descending order based on their relevance to the search query. Finally, please summarize the rank and use [start] to begin the summarization and use [end] to indicate the end of it. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing passages. The format of the summarization should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. In the summarization, only response the ranking results."
    elif prompt_type == 17:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step .... First please summarize each passage. Then rate the relevance between the query and each passage from score 0 to 19 based on the summarization and the original passages. All passages should have distinctive scores. After that, please sort identifiers in descending order based on their relevance to the search query. Finally, please give the rank and use [start] to begin the rank and use [end] to indicate the end of it. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing passages. The format of the rank should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. In the rank, only response the ranking results."
    elif prompt_type == 18:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Please think step by step .... First please extract key words from each passage. Then rate the relevance between the query and each passage from score 0 to 19 based on the key words and the original passages. All passages should have distinctive scores. After that, please sort identifiers in descending order based on their relevance to the search query. Finally, please give the rank and use [start] to begin the rank and use [end] to indicate the end of it. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing passages. The format of the rank should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. In the rank, only response the ranking results."
    elif prompt_type == 19:
        return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. Let's think step by step .... First please explain the reasons why you think the query and each passage are relevant or not. Then rate the relevance between the query and each passage from score 0 to 19 with all passages have distinctive scores. After that, please sort identifiers in descending order based on their relevance to the search query. Finally, please summarize the rank and use [start] to begin the summarization and use [end] to indicate the end of it. All the passages should be included and listed using identifiers and make sure there is no repetitious and missing passages. The format of the summarization should be [Start] [] > [] [End], e.g., [Start] [1] > [2] [End]. In the summarization, only response the ranking results."


def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo'):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300
    while True:
        messages = get_prefix_prompt(query, num)
        rank = 0
        for hit in item['hits'][rank_start: rank_end]:
            rank += 1
            content = hit['content']
            content = content.replace('Title: Content: ', '')
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = ' '.join(content.split()[:int(max_length)])
            messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
            messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
        if prompt_type == 0:
            messages.append({'role': 'user', 'content': get_post_prompt(query, num)})
        else:
            messages.append({'role': 'user', 'content': get_post_cot_prompt(query, num)})

        if num_tokens_from_messages(messages, model_name) <= max_tokens(model_name) - 200:
            break
        else:
            max_length -= 1
    if print_messages == 1:
        print('*'*100)
        print('message: ', messages)
        print('*'*100)
    return messages


def run_llm(messages, api_key=None, model_name="gpt-3.5-turbo"):
    agent = SafeOpenai(api_key)
    response = agent.chat(model=model_name, messages=messages, temperature=0, return_text=True)
    if print_messages:
        print('*'*100)
        print('original response: ', response)
        print('*'*100)
    return response


def clean_response(response: str):
    global current_mismatch_query_cnt
    if prompt_type != 0:
        pattern = r'\[Start\](.*?)\[End\]'
        result = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if result:
            extracted_data = result.group(1)
            response =  extracted_data
        else:
            current_mismatch_query_cnt = 1

    new_response = ''        
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
        
    return new_response


def remove_duplicate(response):
    new_response = []
    current_rep_pasg_cnt = 0
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            current_rep_pasg_cnt+=1
    return new_response, current_rep_pasg_cnt


def receive_permutation(item, permutation, rank_start=0, rank_end=100):   
    global correct_malform, rep_query_cnt, rep_passage_cnt, miss_query_cnt, miss_passage_cnt, mismatch_query_cnt, current_mismatch_query_cnt
    
    current_rep_query_cnt, current_rep_passage_cnt = 0, 0
    current_miss_query_cnt, current_miss_passage_cnt = 0, 0
    current_mismatch_query_cnt = 0
    response = clean_response(permutation)
    if print_messages == 1:
        print('*'*100)
        print('clean response: ', response)
        print('*'*100)
    response = [int(x) - 1 for x in response.split()]
    response, current_rep_passage_cnt = remove_duplicate(response)
    if current_rep_passage_cnt > 0:
        current_rep_query_cnt = 1
        rep_query_cnt += 1
        rep_passage_cnt += current_rep_passage_cnt
    if len(response) < rank_end - rank_start:
        current_miss_query_cnt = 1
        current_miss_passage_cnt = rank_end - rank_start - len(response)
        miss_query_cnt += 1
        miss_passage_cnt += current_miss_passage_cnt
    if current_mismatch_query_cnt > 0:
        mismatch_query_cnt += 1
    print(f'current_rep_passage_cnt: {current_rep_passage_cnt}, current_miss_passage_cnt: {current_miss_passage_cnt}, current_mismatch_query_cnt: {current_mismatch_query_cnt}')
    if correct_malform and (current_rep_passage_cnt > 0 or current_miss_passage_cnt > 0 or current_mismatch_query_cnt > 0):
        print(
            'repetition, missing, or mismatch: \n',
           f'original response:\n {permutation}\n'
           f'original response rank: {clean_response(permutation)}\n',
           f'clean response rank: {response}\n')
        return item, 0
    
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    if print_messages == 1:
        print('*'*100)
        print('original item: ', item)
        print('*'*100)
    if current_rep_passage_cnt > 0 or current_miss_passage_cnt > 0 or current_mismatch_query_cnt > 0:
        print(
            'repetition, missing or mismatch: \n',
           f'original response: {permutation}\n'
           f'original response rank: {clean_response(permutation)}\n',
           f'post response rank: {response}\n')
    return item, 1


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None):
    global chat_cnt
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end,
                                              model_name=model_name) # chan
    flag = 0
    cnt = 0
    while not flag and cnt <= 50:
        permutation = run_llm(messages, api_key=api_key, model_name=model_name)
        chat_cnt += 1
        print(f'chat_cnt: {chat_cnt}')
        if chat_cnt == 1:
            print('*'*100)
            print('Message Example:\n', messages)
            print('Permutation Example:\n', permutation)
            print('*'*100)
        item, flag = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
        cnt+=1
        if cnt <=3:
            messages[-1]['content'] = messages[-1]['content'] + 'Please follow the summarization format and do not miss or repeat any passage.'
    return item


def sliding_windows(args, item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None):
    global prompt_type, print_messages, correct_malform, rep_query_cnt, rep_passage_cnt, miss_query_cnt, miss_passage_cnt, mismatch_query_cnt
    rep_query_cnt, rep_passage_cnt, miss_query_cnt, miss_passage_cnt, mismatch_query_cnt = 0, 0, 0, 0, 0
    prompt_type = args.prompt_type 
    print_messages = args.print_messages
    correct_malform = args.correct_malform
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
        if args.debug:
            break
    print(
        'current query statistics:\n'
       f'rep_query_cnt: {rep_query_cnt}\n',
       f'rep_passage_cnt: {rep_passage_cnt}\n',
       f'miss_query_cnt: {miss_query_cnt}\n',
       f'miss_passage_cnt: {miss_passage_cnt}\n',
       f'mismatch_query_cnt: {mismatch_query_cnt}\n')
    args.rep_query_cnt += rep_query_cnt
    args.rep_passage_cnt += rep_passage_cnt
    args.miss_query_cnt += miss_query_cnt
    args.miss_passage_cnt += miss_passage_cnt
    args.mismatch_query_cnt += mismatch_query_cnt
    if print_messages == 1:
        print('*'*100)
        print('slide window item: ', item)
        print('*'*100)
    return item


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def main():
    from pyserini.search import LuceneSearcher
    from pyserini.search import get_topics, get_qrels
    import tempfile

    api_key = None  # Your openai key

    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    topics = get_topics('dl19-passage')
    qrels = get_qrels('dl19-passage')

    rank_results = run_retriever(topics, searcher, qrels, k=100)

    new_results = []
    for item in tqdm(rank_results):
        new_item = permutation_pipeline(item, rank_start=0, rank_end=20, model_name='gpt-3.5-turbo',
                                        api_key=api_key)
        new_results.append(new_item)

    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(new_results, temp_file)
    from trec_eval import EvalFunction

    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', 'dl19-passage', temp_file])


if __name__ == '__main__':
    main()