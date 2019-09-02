import pandas as pd
from nltk.tokenize import word_tockenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def tockenize(text):
    tockenized_text = []
    tokenizer = word_tockenize()
    
    for w in text:    
        tockenized_text = tokenizer.tokenize(w)
            
    return tockenized_text    

def remove_stopwords(tockenized_words):
    stop_words = set(stopwords.words("english"))
    
    filtered_list = []
    
    for w in tockenized_words:
        if w not in stop_words:
            filtered_list.append(w)
            
    return filtered_list
    
def lemmatize_words(tockenized_words):
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_list = []
    
    for w in tockenized_words:
        lemmatized_list = lemmatizer.lemmatize(w)
    
    return lemmatized_list

def filter_all(column_name):
    filtered_text = []
    for texts in column_name:
        tockenizing_clean = tockenize(texts)
        first_clean = remove_stopwords(tockenizing_clean)
        second_clean = lemmatize_words(first_clean)
        filtered_text = second_clean
    
    return filtered_text
        
    
    
    
    