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
    cols = ['ID', 'Content']
    df = df[cols]
    return df
def articles_2():
    df_new = articles()
    cleaned_df_new = filter_all_2(df_new)
    pd.to_pickle(cleaned_df_new, 'article_dataframe.pkl')
"""
def final_df():
    cleaned_df = preprocess.filter_all()
    cleaned_df['article_content'] = np.nan
    cleaned_df['article_content'] = cleaned_df['article_content'].astype(object)
    final_df_1 = pd.DataFrame(columns = ['ID', 'claim', 'label', 'related_articles', 'article_content'])
    final_df_1.set_index('ID', inplace=True)
    df_new = articles()
    cleaned_df_new = filter_all_2(df_new)
    cleaned_df_new_list = cleaned_df_new.values.tolist()
    for value in cleaned_df['related_articles']:
        article_list = []
        for i in value:
            for list in cleaned_df_new_list:
                list[0] = int(list[0])
                if list[0] == i:
                    article_list.append(list[1])
        article_list_2 = (int(cleaned_df.index[cleaned_df['related_articles'] == value]), list(cleaned_df.loc[cleaned_df.index[cleaned_df['related_articles'] == value], 'claim']), int(cleaned_df.loc[cleaned_df.index[cleaned_df['related_articles'] == value], 'label']), list(value), list(article_list))
        article_list_2 = list(article_list_2)
        final_df_1.loc[len(final_df_1), :] = article_list_2
    return final_df_1
"""

if __name__ == "__main__":
    articles_2()
