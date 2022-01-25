import json
import pandas as pd


def extract_text(json_file):
    summary= []
    sentences = []
    total_length = len(json_file['documents'])
    for index in range(total_length):
        sentence = ''
        docu = json_file['documents'][index]['text']
        docu_len = len(docu)

        abstract = json_file['documents'][index]['abstractive']
        total_abs = ''
        for i in range(len(abstract)):
            total_abs += abstract[i]

        summary.append(total_abs)

        for docu_idx in range(docu_len):
            paragraph = docu[docu_idx]
            para_len = len(paragraph)

            for idx in range(para_len):
                sentence += paragraph[idx]['sentence']
        sentences.append(sentence)
    
    return sentences,summary

def simple_preprocess(text):
    result = ''
    for t in text:
        if t.isalnum() or t == " " or t ==".":
            result += t
    return result

def load_data(file_name,save = True):
    with open(file_name) as file:
        json_file = json.load(file)
    texts,abst = extract_text(json_file)
    
    frame = pd.DataFrame({"abstract" : abst,"sentence" : texts})

    #save data frame(opt.)
    if save:
        frame.to_csv('total_data.csv',encoding = 'utf8')

    return frame
