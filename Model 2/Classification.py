import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
import Json_Read as jr
#import data
df = jr.create_dataframes_claim_label('../TrueNorthAI')

#Clean unnecessary symbols
df['Claim_Parsed_1'] = df['claim'].str.replace("\r", " ")
df['Claim_Parsed_1'] = df['Claim_Parsed_1'].str.replace("\n", " ")
df['Claim_Parsed_1'] = df['Claim_Parsed_1'].str.replace("    ", " ")
df['Claim_Parsed_1'] = df['Claim_Parsed_1'].str.replace('"', '')

#lowercase everything
df['Claim_Parsed_2'] = df['Claim_Parsed_1'].str.lower()

#clear out punctuation
punctuation_signs = list("?:!.,;")
df['Claim_Parsed_3'] = df['Claim_Parsed_2']
for punct_sign in punctuation_signs:
    df['Claim_Parsed_3'] = df['Claim_Parsed_3'].str.replace(punct_sign, '')

#No possessive pronouns
df['Claim_Parsed_4'] = df['Claim_Parsed_3'].str.replace("'s", "")
print("1 Complete")
#Lemmatize all the words
wordnet_lemmatizer = WordNetLemmatizer()
nrows = len(df)
lemmatized_text_list = []
for row in range(0, nrows):
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    # Save the text and its words into an object
    j = df.index.values.tolist()
    for value in j:
        text = df.loc[value]['Claim_Parsed_4']
        text_words = text.split(" ")
    # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)
df['Claim_Parsed_5'] = lemmatized_text_list
print("2 complete")
#getting rid of stopwords
stop_words = list(stopwords.words('english'))
df['Claim_Parsed_6'] = df['Claim_Parsed_5']
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Claim_Parsed_6'] = df['Claim_Parsed_6'].str.replace(regex_stopword, '')

#clean up the dataframe
list_columns = ["claim", "Claim_Parsed_6", "label"]
df = df[list_columns]
df = df.rename(columns={'Claim_Parsed_6': 'Claim_Parsed'})

X_train, X_test, y_train, y_test = train_test_split(df['Claim_Parsed'], df['label'], test_size=0.20, random_state=8)

ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=ngram_range, stop_words=None, lowercase=False, max_df=max_df, min_df=min_df, max_features=max_features, norm='l2', sublinear_tf=True)

features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    
    
#https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb
