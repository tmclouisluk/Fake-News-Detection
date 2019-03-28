import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import random


def read_data(path):
    return pd.read_csv(path, index_col=False)


def clean(data):
    def clean_text(x):
        text = ''.join([i for i in x if not i.isdigit()])
        text = re.sub("[!\?@^&.,/#$+%*:()'\"-]", ' ', text)
        return text

    def clean_stopwords(x):
        text = x.split()
        filtered_words = [word.lower() for word in text if word not in stopwords.words('english')]
        return filtered_words

    data['cleaned_statement'] = data.loc[:, 'Statement'].apply(clean_text)
    data['text_array'] = data.loc[:, 'cleaned_statement'].apply(clean_stopwords)


def tokenizer(data):
    word_to_ix = {}
    total_text_array = []
    for text_array in data['text_array']:
        for word in text_array:
            total_text_array.append(word)

    random.shuffle(total_text_array)

    for word in total_text_array:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    def text_tokenizer(x):
        tokens = []
        for text in x:
            token = word_to_ix[text]
            tokens.append(token)
        return tokens

    data['text_token'] = data.loc[:, 'text_array'].apply(text_tokenizer)

    return word_to_ix


def one_hot_encoding(data):
    label_to_ix = {"True": 1, "False": 0}

    def label_one_hot(x):
        return 1 if x else 0

    data['label_one_hot'] = data.loc[:, 'Label'].apply(label_one_hot)

    return label_to_ix
