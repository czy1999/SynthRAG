import numpy as np
import time
from openai import OpenAI
import pandas as pd
import concurrent.futures
import dashscope
import os
# 设置 dashscope API 密钥
API_KEY = os.getenv("QWEN_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

df = pd.read_csv('./data/zhihu_filter.csv', index_col=0)
corpus_emb = np.load('./data/corpus_emb.npy')
client = OpenAI(api_key= API_KEY,base_url=BASE_URL)

def call_with_prompt_stream(query, search_enhance=True):
    response_generator = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        stream=True,
        extra_body={"enable_search": search_enhance}
    )
    return response_generator

def call_with_prompt(prompt,search_enhance = True):
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': prompt}],
        extra_body={"enable_search":search_enhance})
    return completion.choices[0].message.content

def embed_with_str(x):
    resp = dashscope.TextEmbedding.call(
        api_key= API_KEY,
        model=dashscope.TextEmbedding.Models.text_embedding_v2,
        input=x
    )
    return resp['output']['embeddings'][0]['embedding']


def encode_text(text):
    embedding = embed_with_str(text)
    return text, embedding  


def get_sim_index(query_emb,corpus_emb):
    dot_product = np.dot(corpus_emb, query_emb)
    query_emb_norm = np.linalg.norm(query_emb)
    corpus_emb_norms = np.linalg.norm(corpus_emb, axis=1)
    cosine_similarities = dot_product / (query_emb_norm * corpus_emb_norms)
    sorted_indices = np.argsort(cosine_similarities)[::-1]
    return sorted_indices.tolist()


def sematic_search(query, k = 1):
    query_emb = np.array(embed_with_str(query))
    top_k_indices = get_sim_index(query_emb,corpus_emb)[:k]
    output = {}
    for idx in top_k_indices:
        question = df['问'].iloc[idx]
        new_df = df[df['问'] == question]
        output[question] = new_df['答'].unique().tolist()
    
    return output

def md_process(md):
    # Remove specific headers
    headers_to_remove = ['引言', '引入', '开场白', '正文']
    for header in headers_to_remove:
        md = md.replace(f'## {header}', '').replace(f'# {header}', '')
    
    # Split into lines and filter out conclusion-like statements
    conclusion_starters = ['综上所述', '总', '结']
    lines = md.split('\n\n')
    filtered_lines = [line for line in lines[1:] if not any(line.startswith(starter) for starter in conclusion_starters)]
    
    # Join the filtered lines back into a string
    return '\n\n'.join(filtered_lines)


def sematic_search_v2(query, k=10, max_len=5000):
    """
    Use semantic search model to find the most relevant questions and their responses.

    Args:
       query (str): Input query text
       df (pandas.DataFrame, optional): DataFrame storing questions and answers, default is None

    Returns:
       dict: Dictionary containing the most relevant questions and their responses
    """
    
    query_emb = np.array(embed_with_str(query))
    top_k_indices = get_sim_index(query_emb, corpus_emb)[:k]
    
    output = {}
    for idx in top_k_indices:
        question = df['问'].iloc[idx]
        new_df = df[df['问'] == question]
        responses = new_df['答'].unique().tolist()
        
        # 截断答复以保证不超过最大长度
        truncated_responses = []
        for response in responses:
            if len(response) < max_len:
                output[question] = response
    
    return output


def multi_thread_query(sub_queries):
    """
    Use multi-threading to call the call_with_prompt function in parallel,
    process a series of queries and return a dictionary of results.

    Args:
       sub_queries (list): List containing a series of query texts

    Returns:
       dict: Dictionary containing the results for each query, with query text as keys and corresponding response results as values
    """
    results_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_query = {}
        for query in sub_queries:
            time.sleep(1)
            future = executor.submit(call_with_prompt, query)
            future_to_query[future] = query

        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                response = future.result()
                results_dict[query] = response.output.text
            except Exception as e:
                print(f"Failed: {e}, Query: {query}")
                results_dict[query] = None  
    return results_dict
