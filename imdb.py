import pandas as pd, numpy as np, sys, nltk, string, h5py
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

class_names = ['negative', 'positive']
words_in_model = 5000
max_doc_length = 500
embedding_vector_len = 32


def _tokenize(doc, stemmer=None):
    tokens = nltk.word_tokenize(doc)
    tokens = [i for i in tokens if i not in string.punctuation]
    return [stemmer.stem(t) if stemmer else t for t in tokens]


def _preprocess(docs, word_dict, max_doc_len, stemmer=None):
    padding_val = 0.
    unknown_word_idx = 1
    word_dict_idx_offset = 2

    encoded_doc = [[word_dict[t] + word_dict_idx_offset if t in word_dict else unknown_word_idx
                    for t in _tokenize(doc, stemmer)]
                   for doc in docs]

    # truncate and pad input sequences
    return sequence.pad_sequences(encoded_doc, maxlen=max_doc_len, value=padding_val)


def preprocess_and_save(max_doc_len=500, words_in_model=5000):
    print 'loading data'
    df = pd.DataFrame.from_csv('labeledTrainData.tsv', sep='\t', encoding='utf-8')
    X, y = [df[col].values for col in ['review', 'sentiment']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print 'tokenizing and stemming'
    stemmer = SnowballStemmer('english')
    words = [token for doc in X_train for token in _tokenize(doc, stemmer)]

    print 'calc word freq distributions'
    fdist = nltk.FreqDist(words)
    word_dict = {w: idx for idx, (w, _) in enumerate(fdist.most_common(words_in_model))}

    print 'preprocessing train and test sets'
    X_train_enc, X_test_enc = map(lambda ds: _preprocess(ds, word_dict, max_doc_len, stemmer),
                                  [X_train, X_test])

    with h5py.File('train_test.h5', 'w') as hf:
        hf.create_dataset('X_train', data=X_train_enc)
        hf.create_dataset('X_test', data=X_test_enc)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('y_test', data=y_test)


def load_train_test():
    with h5py.File('train_test.h5', 'r') as hf:
        return map(lambda ds: np.array(hf.get(ds)),
                   ['X_train', 'X_test', 'y_train', 'y_test'])


def train():
    X_train, X_test, y_train, y_test = load_train_test()
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    model = Sequential()
    model.add(Embedding(words_in_model + 2, embedding_vector_len, input_length=max_doc_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)

    model.save_weights("model.h5")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'feat':
        preprocess_and_save(max_doc_len=max_doc_length, words_in_model=words_in_model)
    elif cmd == 'train':
        train()
    else:
        raise Exception('unknown command ' + cmd)
