import pandas as pd
import os

def create_dataframes(path_to_TrueNorthAI):
    path = path_to_TrueNorthAI
    df1 = pd.read_json(path+"/train.json")
    df1.set_index('id', inplace=True)
    cols = df1.columns.tolist()
    cols = ['claim', 'claimant', 'date', 'related_articles', 'label']
    df1 = df1[cols]
    print(df1)
    
def create_dataframes2(path_to_TrueNorthAI):
    path = path_to_TrueNorthAI
    article_id = []
    article_content = []
    for filename in os.listdir(path+"/train_articles/"):
        article_id.append(filename.split('.')[0])
        with open(path+"/train_articles/"+filename, encoding="utf8") as f_input:
            article_content.append(f_input.read())
            print("article complete")
    
    data = {'ID': article_id, 'Content': article_content}
    df2 = pd.DataFrame(data)
    df2.set_index('ID', inplace=True)
    print(df2)   

if __name__ == "__main__":
    create_dataframes('True North AI')
    create_dataframes2('True North AI')