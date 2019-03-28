from data_processing import read_data, clean, tokenizer, one_hot_encoding
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    TRAIN_CSV_PATH = "./data/train.csv"
    MAX_SEQUENCE_LENGTH = 20

    data = read_data(TRAIN_CSV_PATH)

    clean(data)
    word_to_ix = tokenizer(data)
    label_to_ix = one_hot_encoding(data)

    def trim_zero_padding(x):
        arr = x[:MAX_SEQUENCE_LENGTH]
        arr = arr + [0] * (MAX_SEQUENCE_LENGTH - len(arr))
        return arr

    data['text_token'] = data.loc[:, 'text_token'].apply(trim_zero_padding)
    X = list(data['text_token'])
    y = list(data['label_one_hot'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    svm = SVC(kernel="rbf", random_state=1, gamma=0.2, C=1.0)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)


if __name__ == "__main__":
    main()

