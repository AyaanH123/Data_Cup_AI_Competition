import Preprocessing as preprocess
import os
import pickle
import pandas as pd
import numpy as np

def articles():
    article_id = []
    article_content = []
    for filename in os.listdir('../train_articles/'):
        article_id.append(filename.split('.')[0])
        with open('../train_articles/' + filename, encoding="utf8") as f_input:
            article_content.append(f_input.read())
    data = {'ID': article_id, 'claim': article_content}
    df = pd.DataFrame(data)
    df.set_index('ID', inplace=True)
    return df
def filter_all_2(df):
    df = preprocess.symbols(df)
    df = preprocess.lowercasing(df)
    df = preprocess.punctuation(df)
    df = preprocess.possession(df)
    filtered_text = []
    my_list = df['claim'].values
    for texts in my_list:
        tokenizing_clean = preprocess.tokenize(texts)
        #first_clean = lemmatize_words(tokenizing_clean)
        second_clean = preprocess.remove_stopwords(tokenizing_clean)
        filtered_text.append(second_clean)
    df['Content'] = filtered_text
    cols = ['Content']
    df = df[cols]
    return df
def final_df():
    cleaned_df = preprocess.filter_all()
    cleaned_df['article_content'] = np.nan
    df_new = articles()
    cleaned_df_new = filter_all_2(df_new)
    for value in cleaned_df['related_articles']:
        article_list = []
        for i in value:
            article_list.append(cleaned_df_new.loc[i, ['related_articles']])
        article_list =  " ||| ".join(article_list)
        cleaned_df.at[cleaned_df.index[cleaned_df['related_articles'] == value], 'article content'] = article_list
    return cleaned_df

if __name__ == "__main__":
    df_final = final_df()
    with open('final_dataframe', 'wb') as output:
        pickle.dump(df_final, output)