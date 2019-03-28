from data_processing import read_data, clean, tokenizer, one_hot_encoding
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    TRAIN_CSV_PATH = "./data/train.csv"
    MAX_SEQUENCE_LENGTH = 20

    data = read_data(TRAIN_CSV_PATH)

    clean(data)
    word_to_ix = tokenizer(data)
    label_to_ix = one_hot_encoding(data)

    #X = np.zeros(b.shape)list(data['text_token'])
    y = list(data['label_one_hot'])


if __name__ == "__main__":
    main()

