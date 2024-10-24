import json
import re
import numpy as np
import pandas as pd
import concurrent.futures
import time
from utils import *  
import pickle
from prompt_template import *
from collections import defaultdict
import copy
import json

class SynthRAG:
    def __init__(self, kmeans_model_path='./data/kmeans_v1.pkl', outline_path = './data/outlines.csv'):
        with open(kmeans_model_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        self.outline_dict = pd.read_csv(outline_path)
        self.details = defaultdict(dict)
        self.few_shot_num = 3
        self.max_example_len = 6000
        self.max_workers = 10
        self.query = ''
        
    def process(self,query, save_file = True,MM = False):
        self.query = query
        outline_text,outline_dict = self.generate_outline(query)
        print('Outline down.')
        self.threaded_process_dict(outline_dict, query, outline_text, max_workers=self.max_workers)
        print('Text dwon.')
        self.details[query]['output_dict'] = copy.deepcopy(outline_dict)
        if MM:
            self.threaded_process_dict_mm(outline_dict,query)
            print('Image down.')
            self.details[query]['output_dict_MM'] = outline_dict
        markdown_output = self.generate_markdown(outline_dict)
        self.details[query]['output'] = markdown_output
        if save_file:
            self.save_markdown(markdown_output, './data/output/{}_v3.md'.format(query))

    def process_short(self,query, save_file = True,MM = False):
        self.query = query
        outline_text,outline_dict = self.generate_outline(query)
        print('Outline down.')
        prompt_answer = self.generate_short_answer(query, outline_text)
        print('Text dwon.')
        self.details[query]['output'] = prompt_answer
        if save_file:
            self.save_markdown(prompt_answer, './data/output/short_{}_v1.md'.format(query))
        return prompt_answer

    def generate_and_save_samples(self, qa, outline_prompt, topnum = 4, downnum = 2, filename='output.jsonl'):
        
        def get_top_bottom_questions(df, label, topnum, downnum):
            filtered_df = df[df['label'] == label]
            top_questions = filtered_df.nlargest(topnum, 'voteup_count')[['question', 'content']]
            bottom_questions = filtered_df.nsmallest(downnum, 'voteup_count')[['question', 'content']]
            # 返回结果
            return {
                '高质量样例': top_questions,
                '低质量样例': bottom_questions
            }
        
        def format_questions_to_text(self, questions_data):
            formatted_text = ""
            for kind, questions in questions_data.items():
                formatted_text += f"--- {kind.replace('_', ' ').title()}开始---\n"
                for index, row in questions.iterrows():
                    q_title = row['question']
                    content = row['content']
                    formatted_text += f"问题: {q_title}\n回答: {content}\n\n"
                formatted_text += f"--- {kind.replace('_', ' ').title()}结束 ---\n\n\n"
            return formatted_text
        
        
        
        samples = []

        for i in qa.label.unique():
            j = 1
            original_result = get_top_bottom_questions(qa, label=i, topnum=topnum, downnum=downnum)
            formatted_result = format_questions_to_text(original_result)
            p = outline_prompt.replace('{回答样本}', formatted_result)

            while len(p) > 60000:
                topnum_temp = topnum - j
                downnum_temp = downnum - j
                original_result = get_top_bottom_questions(qa, label=i, topnum=topnum_temp, downnum=downnum_temp)
                formatted_result = format_questions_to_text(original_result)
                p = outline_prompt.replace('{回答样本}', formatted_result)
                j += 1

            id = 'cluster_' + str(i) + '_top4_down2_v2'
            sample = {"data": {"model": "gpt-4-0125-preview", "messages": [{'role': 'user', 'content': p}], "n": 1}, "id": id}
            samples.append(sample)

        # 保存为.jsonl文件
        with open(filename, 'w', encoding='utf-8') as outfile:
            for sample in samples:
                json_str = json.dumps(sample, ensure_ascii=False)  # 确保中文字符不会被ASCII编码
                outfile.write(json_str + '\n')  # 写入文件并添加换行符

    def parse_outline_file(self, file_path):
        """
        Parse a JSON lines file and extract question-answer pairs from it.

        Args:
        - file_path: A string representing the path to the file that contains JSON lines.

        Returns:
        - outline_dict: A dictionary with keys as the extracted IDs and values as dictionaries containing 'q' and 'a'.
        """
        outline_dict = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        json_object = json.loads(line.strip())
                        id_match = re.search(r'cluster_(\d+)_top4_down2', json_object['id'])
                        if id_match:
                            id = id_match.group(1)
                            q = json_object['data']['messages'][0]['content']
                            a = json_object['answer']['choices'][0]['message']['content']
                            outline_dict[id] = {'q': q, 'a': a}
                    except json.JSONDecodeError:
                        print("Error decoding JSON from line:", line)
                    except (KeyError, IndexError):
                        print("Error accessing data within JSON object:", json_object)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        return outline_dict


    def get_similiar_outline(self,query):
        query_emb = np.array(embed_with_str(query))
        cluster = self.kmeans.predict([query_emb])[0]
        out = self.outline_dict[self.outline_dict['cluster'] == str(cluster)]

        if out.shape[0] > 0:
            outline_sample = out['a'].values[0]
            self.details[query]['outline_sample_sucess'] = True
        else:
            outline_sample = self.outline_dict.iloc[0]['a']
            self.details[query]['outline_sample_sucess'] = False

        self.details[query]['outline_sample'] = outline_sample
        return outline_sample

    
    def get_similiar_examples(self,query, k =3):
        sim_res = sematic_search_v2(query)
        self.details[query]['sim_questions'] = sim_res
        example = ''
        num_pairs_selected = 0
        for question, answer in sim_res.items():
            if num_pairs_selected < k:
                pair = '样例问题:' + question + '\n' + '样例回答:' + answer + '\n'
                if len(example) + len(pair) <= self.max_example_len:
                    example += pair
                    num_pairs_selected += 1
                else:
                    break
        self.details[query]['sim_question_texts'] = example
        return example


    def generate_outline_prompt(self, query, k=1, Max_len=5000):
        outline_sample = self.get_similiar_outline(query)
        examples = self.get_similiar_examples(query,self.few_shot_num)
       
        prompt = outline_generate_prompt.replace('{outline_instruct}', outline_sample)\
                                        .replace('{examples}', examples)\
                                        .replace('{query}', query)
        self.details[query]['outline_prompt'] = prompt

        return prompt, examples, outline_sample
    
    def markdown_to_json(self,markdown_text):

        pattern = re.compile(r'(#+) (.+)')
        lines = markdown_text.strip().split('\n')
        json_structure = {}
        current_section = json_structure

        for line in lines:
            match = pattern.match(line)
            if match:
                level = len(match.group(1)) 
                title = match.group(2).strip() 
                content = []

                if level == 1:
                    current_section = json_structure[title] = {}
                elif level == 2:
                    current_section = json_structure[last_level1_title][title] = {}
                elif level == 3:
                    current_section = json_structure[last_level1_title][last_level2_title][title] = {}

                if level == 1:
                    last_level1_title = title
                elif level == 2:
                    last_level2_title = title
            else:
                if line.strip() != '':
                    if '内容' not in current_section:
                        current_section['内容'] = line.strip()
                    else:
                        current_section['内容'] += '\n' + line.strip()

        return json_structure
    
    def generate_outline(self, query, k=1, Max_len=5000):

        # 获取大纲提示
        prompt, example, outline_sample = self.generate_outline_prompt(query, k, Max_len)
        outline_text = call_with_prompt(prompt).output.text
        
        self.details[query]['outline_text'] = outline_text
        
        outline_dict = self.markdown_to_json(outline_text)
        self.details[query]['outline_dict'] = copy.deepcopy(outline_dict)
        
        return outline_text,outline_dict
  

    def generate_markdown_v2(self, d, level=1):
        markdown_content = ""
        for key, value in d.items():
            if isinstance(value, dict) and '内容' in value:
                if level == 1:
                    markdown_content += f"{'#' * level} {key}\n\n"
                markdown_content += f"{value['内容']}\n\n"
            elif isinstance(value, dict):
                if level == 1:
                    markdown_content += f"{'#' * level} {key}\n\n"
                markdown_content += self.generate_markdown(value, level)
        return markdown_content

  
    def generate_markdown(self, d, level=1):
        markdown_content = ""
        for key, value in d.items():
            if isinstance(value, dict) and '内容' in value:
                markdown_content += f"{'#' * level} {key}\n\n"
                markdown_content += f"{value['内容']}\n\n"
            elif isinstance(value, dict):
                markdown_content += f"{'#' * level} {key}\n\n"
                markdown_content += self.generate_markdown(value, level + 1)
        return markdown_content

    def save_markdown(self, markdown_output, filename="output.md"):
        """
        Saves a string containing Markdown content to a file with the specified filename.

        :param markdown_text: The Markdown content to be saved.
        :param filename: The filename for the .md file (should include .md extension).
        """
        if not filename.endswith('.md'):
            filename += '.md'

        with open(filename, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_output)

    def clean_and_export_outlines(self, input_file_path, output_file_path, min_length=200):
        # Function logic here
        pass
    
    def get_keywords(self,query):
        # 获取关键词
        prompt = keywords_prompt.replace('{query}',query)
        res = call_with_prompt(prompt)
        return res

    def generate_short_answer(self, query,outline_text, max_workers=10,return_prompt = False):
        
        example = self.details[query]['sim_question_texts']
        prompt = sub_query_prompt_short.replace('<query>',query).replace('<outline>',outline_text).replace('<style_example>',example)
        if return_prompt:
            return prompt
        else:
            res = call_with_prompt(prompt)
            return res

    def threaded_process_dict(self, d, query,outline_text, max_workers=10):
        
        example = self.details[query]['sim_question_texts']
        
        def process_content(content):
            prompt = sub_query_prompt.replace('<query>',query).replace('<sub_query>',content).replace('<outline>',outline_text).replace('<style_example>',example)
            res = call_with_prompt(prompt)
            return res

        def process_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    process_dict(value)
                elif key == '内容':
                    d[key] = process_content(value)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_dict, item): item for item in d.values()}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print('Generated an exception: %s' % (exc))

    def threaded_process_dict_with_style(self, d, query,outline_text, max_workers=10):
        
        example = self.details[query]['sim_question_texts']
        
        def process_content(content):
            prompt = sub_query_prompt.replace('<query>',query).replace('<sub_query>',content).replace('<outline>',outline_text).replace('<style_example>',example)
            # print(prompt + '--'*10)
            res = call_with_prompt(prompt)
            return res

        def process_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    process_dict(value)
                elif key == '内容':
                    d[key] = process_content(value)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_dict, item): item for item in d.values()}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print('Generated an exception: %s' % (exc))      

                    
    def save_dict_to_file(self,file):
        with open(file, 'w') as file:
            json.dump(self.details, file, ensure_ascii=False, indent=4)
            
    def reset(self):
        self.details = defaultdict(dict)
        self.query = ''
    
