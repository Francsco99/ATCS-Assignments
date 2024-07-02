import json
import re
import pandas as pd
import csv
from collections import Counter
import os
from concurrent.futures import ThreadPoolExecutor

def lowercase(text):
    return text.lower()

def remove_useless_words(text):
    # remove words containing numbers followed by units of measurement
    text = re.sub(r'\b\d+\s*(?:\.\d+)?(?:inch|in|cm|mm|ms|dc|kg|gb|g?hz|days?|months?|years?)\s*\b', '', text)

    # remove words representing resolutions (like 1920x1080)
    text = re.sub(r'\b\d+\s*x\s*\d+\b', '', text)

    # remove frequent words
    text = re.sub(r'\b(?:led|lcd|monitor|vga|hdmi|led|(windows|win)\s*(\d{1,2}|vista|xp)*|pixel|display|touch|touchscreen|touchmonitor|screen|widescreen|tft|tv|cable|port|desktop|ios|apple|ebay|new|shop|free|item|colou?r)s?', '', text)

    return text

def remove_url(text):
    return re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '', text)

def remove_white_numbers(text):
     return re.sub(r'(\d)\s+(\d)', r'\1\2', text)

def remove_extra_white(text):
     return ' '.join(text.split())

def replace_special_withespace(text,to_exclude: list[str]=[]):
    return ''.join(c if c.isalnum() or c in to_exclude else ' ' for c in text)

def find_alfanum(text):
    words = re.findall(r'\w+', text)
    return [word for word in words if re.search(r'[A-Za-z]', word) and re.search(r'\d', word)]

def clean_text(text):
    text = lowercase(text)
    text = remove_url(text)
    text = replace_special_withespace(text)
    text = remove_useless_words(text)
    text = remove_white_numbers(text)
    text = remove_extra_white(text)
    return text

def get_raw_model_name(data):
    relevant_labels = ['model', 'model name', 'product model', 'model number', 'product name', 'part', 'product description']

    for label in relevant_labels:
        if data.get(label):
            if isinstance(data[label], list):
                return data[label][0]
            return data[label]
    
    return data['<page title>']

def get_page_title(data):
    return data['<page title>']

def find_model_name(text):
    raw_model = find_alfanum(text)
    model = ' '.join(word if len(word) > 3 else '' for word in raw_model)
    model = remove_extra_white(model)
    return model 

def remove_common_words(source_path,item_to_title,word_counter, threshold = 0.02):
    min_occur = len(item_to_title) * threshold

    remove_words=[]
    for word, count in word_counter.most_common():
        if count < min_occur: continue
        remove_words.append(word)

    print(f"parole da rimuovere da {source_path}: {remove_words}")

    for k,v in item_to_title.items():
        for word in v.split():
            if word not in remove_words: continue
            v = v.replace(word,'')
            v = remove_extra_white(v)
            item_to_title[k]=v


def process_sources(root_path,sources_path):
    item_to_title={}
    common_words_counter=Counter()
    for k, v, files in os.walk(root_path + '/' + sources_path):
        for file in files:
            file_path = os.path.join(root_path+'/'+sources_path,file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data is not None:
                        item = sources_path + '//' + file.replace('.json','')
                        raw_text = get_raw_model_name(data)
                        proc_text = clean_text(raw_text)
                        if len(proc_text.split()) > 1:
                            model_name = find_model_name(proc_text)
                            if model_name !='':
                                proc_text=model_name
                        common_words_counter.update(proc_text.split())
                        item_to_title[item]=proc_text
                    else:
                        print(f"File json non esistente o vuoto: {file_path}")
            except Exception as e:
                print(f"Errore nella apertura o lettura di {file_path}: {e}")
            remove_common_words(sources_path,item_to_title,common_words_counter)

    return item_to_title

def process_sources_title(root_path,sources_path):
    item_to_title={}

    for k, v, files in os.walk(root_path + '/' + sources_path):
        for file in files:
            file_path = os.path.join(root_path+'/'+sources_path,file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data is not None:
                        item = sources_path + '//' + file.replace('.json','')
                        title = get_page_title(data)
                        item_to_title[item]=title
                    else:
                        print(f"File json non esistente o vuoto: {file_path}")
            except Exception as e:
                print(f"Errore nella apertura o lettura di {file_path}: {e}")
    
    return item_to_title

def process_sources_main(root_path):
    item_to_title = {}
    processed = {}

    for k, dirs, v in os.walk(root_path):
        for dir in dirs:
            processed.update(process_sources(root_path, dir))
            item_to_title.update(process_sources_title(root_path, dir))

    try:
        proc_path = os.path.join('preprocessing', 'preprocessed_dataset.json')
        with open(proc_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=4)
        print(f"Written to {proc_path}")
    except Exception as e:
        print(f"Error writing {proc_path}: {e}")

    try:
        item_title_path = os.path.join('preprocessing', 'item_to_title.json')
        with open(item_title_path, 'w', encoding='utf-8') as f:
            json.dump(item_to_title, f, ensure_ascii=False, indent=4)
        print(f"Written to {item_title_path}")
    except Exception as e:
        print(f"Error writing {item_title_path}: {e}")
    

process_sources_main('monitor/monitor_specs')
