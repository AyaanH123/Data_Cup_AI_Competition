import pandas as pd
import os
import json

def create_dataframes(path_to_TrueNorthAI):
    path = path_to_TrueNorthAI
    df = pd.read_json(path+"/train.json")
    df.set_index('id', inplace=True)
    cols = df.columns.tolist()
    cols = ['claim', 'claimant', 'date', 'related_articles', 'label']
    df = df[cols]
    return df

def create_dataframes_claim_label(path_to_TrueNorthAI):
    path = path_to_TrueNorthAI
    with open(path+"/train.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)
    cols = df.columns.tolist()
    cols = ['claim', 'label']
    df = df[cols]
    return df

#For input data
def create_dataframes_input(path_to_TrueNorthAI):
    path = path_to_TrueNorthAI
    with open(path+'/metadata.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)
    cols = df.columns.tolist()
    cols = ['claim']
    df = df[cols]
    return df
    
def create_dataframes2(path_to_TrueNorthAI):
    path = path_to_TrueNorthAI
    article_id = []
    article_content = []
    for filename in os.listdir(path+"/train_articles/"):
        article_id.append(filename.split('.')[0])
        with open(path+"/train_articles/"+filename, encoding="utf8") as f_input:
            article_content.append(f_input.read())
    data = {'ID': article_id, 'Content': article_content}
    df = pd.DataFrame(data)
    df.set_index('ID', inplace=True) 
    return df