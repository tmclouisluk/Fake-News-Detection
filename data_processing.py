import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

TRAIN_CSV_PATH = "./data/train.csv"
data = pd.read_csv(TRAIN_CSV_PATH, index_col=False)
data.head(3)


def clean_text(x):
    text = ''.join([i for i in x if not i.isdigit()])
    text = re.sub("[!@#$+%*:()'-]", ' ', text)
    return text


data['cleaned_statement'] = data.loc[:,'Statement'].apply(clean_text, field="Statement")