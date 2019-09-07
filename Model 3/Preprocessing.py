import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import pickle

def import_data():
    with open('../train.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.set_index('id', inplace=True)
    cols = df.columns.tolist()
    cols = ['claim', 'label', 'related_articles']
    df = df[cols]
    return df
def symbols(df):
    df['claim'] = df['claim'].str.replace("\r", " ")
    df['claim'] = df['claim'].str.replace("\n", " ")
    df['claim'] = df['claim'].str.replace("    ", " ")
    df['claim'] = df['claim'].str.replace('"', ' ')
    df['claim'] = df['claim'].str.replace("'", "")
    df['claim'] = df['claim'].str.replace('-', ' ')
    df['claim'] = df['claim'].str.replace('#', ' ')
    return df

def lowercasing(df):
    df['claim'] = df['claim'].str.lower()
    return df

def punctuation(df):
    punctuation_signs = list('"(?:!.,;")')
    df['claim'] = df['claim']
    for punct_sign in punctuation_signs:
        df['claim'] = df['claim'].str.replace(punct_sign, '')
    return df

def possession(df):
    df['claim'] = df['claim'].str.replace("'s", "")
    return df

def tokenize(text):
    tokenized_text = word_tokenize(text)
    return tokenized_text

def remove_stopwords(tokenized_words):
    stop_words = set(stopwords.words("english"))
    filtered_list = []
    for w in tokenized_words:
        if w not in stop_words:
            filtered_list.append(w)
    return filtered_list

def lemmatize_words(tokenized_words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for w in tokenized_words:
        lemmatized_list.append(lemmatizer.lemmatize(w))
    return lemmatized_list

def filter_all():
    df = import_data()
    df = symbols(df)
    df = lowercasing(df)
    df = punctuation(df)
    df = possession(df)
    filtered_text = []
    my_list = df['claim'].values
    for texts in my_list:
        tokenizing_clean = tokenize(texts)
        #first_clean = lemmatize_words(tokenizing_clean)
        second_clean = remove_stopwords(tokenizing_clean)
        filtered_text.append(second_clean)
    df['claim'] = filtered_text
    return df


if __name__ == "__main__":
    cleaned_df = filter_all()
    pd.to_pickle(cleaned_df, 'train_dataframe.pkl')
