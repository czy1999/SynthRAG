# app.py
import streamlit as st
from utils import call_with_prompt_stream
from model import SynthRAG
import pickle
import copy

# Page configuration
st.set_page_config(page_title="SynthRAG", page_icon="./asset/synthrag.png", layout="wide")

# Title
st.title("SynthRAG",)

# Sidebar
st.sidebar.title("Function Selection")
show_prompt = st.sidebar.checkbox("Show prompt", value=False, key="show_sidebar")
search_enhance = st.sidebar.checkbox("Enable search enhancement (By Qwen API)", value=True)

k = st.sidebar.selectbox(
    'Select number of examples',
    options=[2, 3, 4, 5],
    index=0
)

# User input
query = st.text_area("Enter your question:", 
                     value="朝鲜决定解散「祖国战线中央委员会」，释放了什么信号？有何深意？")

# Answer mode selection
mode = st.selectbox("Select answer mode", 
                    ["Direct answer", 'SynthRAG framework generation'],
                    index=0)

# Generate answer
if st.button("Generate answer!"):
    if mode == "Direct answer":
        st.write('Generating answer...')
        response_generator = call_with_prompt_stream(query, search_enhance=search_enhance)
        output_text = st.empty()
        full_response = ""
        for chunk in response_generator:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                output_text.success(full_response + "▉")
        output_text.success(full_response) 
        st.write('Answer completed')

    elif mode == "SynthRAG framework generation":
        st.write('SynthRAG generating answer...')
        syn = SynthRAG()
        syn.max_workers = 5
        syn.query = query
        
        keywords = syn.get_keywords(query).strip()
        st.write(f'Keywords: {keywords}')

        st.write('Generating outline...')
        prompt, examples, outline_sample = syn.generate_outline_prompt(query)
        response_generator = call_with_prompt_stream(prompt, search_enhance=search_enhance)
        outline_text = st.empty()
        full_response = ""
        for chunk in response_generator:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                outline_text.success(full_response + "▉")
        outline_text.success(full_response) 
        outline_text = full_response
        st.write('Outline generation completed!')

        syn.details[query]['outline_text'] = outline_text
        outline_dict = syn.markdown_to_json(outline_text)
        syn.details[query]['outline_dict'] = copy.deepcopy(outline_dict)
        st.write('Outline parsed, starting to write the answer, it will take 2-3 minutes...') 
        syn.threaded_process_dict(outline_dict, query, outline_text, max_workers=10)
        st.write('Text generation completed.')
        syn.details[query]['output_dict'] = copy.deepcopy(outline_dict)
        markdown_output = syn.generate_markdown(outline_dict)
        syn.details[query]['output'] = markdown_output

        markdown_output = md_process(markdown_output)

        syn.save_markdown(markdown_output, f'./output/{query[:40]}.md')

        st.download_button(
            label="Download answer as markdown file",
            data=markdown_output,
            file_name=f'{query}.md',
            mime='text/csv',
        )
        with open(f'./output/{query[:40]}.pkl', 'wb') as f:
            pickle.dump(syn.details[query], f)




