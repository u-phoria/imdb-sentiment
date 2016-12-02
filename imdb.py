import pandas as pd, numpy as np, sys, os, nltk, string
import h5py, json
from keras.engine import Input
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution1D, GlobalMaxPooling1D, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

class_names = ['negative', 'positive']
words_in_model = 5000
max_doc_length = 500


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


def preprocess_and_save(max_doc_len=500, words_in_model=5000, stem=False):
    print 'loading data'
    df = pd.DataFrame.from_csv('labeledTrainData.tsv', sep='\t', encoding='utf-8')
    X, y = [df[col].values for col in ['review', 'sentiment']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print 'tokenizing and stemming'
    stemmer = SnowballStemmer('english') if stem else None
    words = [token for doc in X_train for token in _tokenize(doc, stemmer)]

    print 'calc word freq distributions'
    fdist = nltk.FreqDist(words)
    word_dict = {w: idx for idx, (w, _) in enumerate(fdist.most_common(words_in_model))}

    print 'preprocessing train and test sets'
    X_train_enc, X_test_enc = map(lambda ds: _preprocess(ds, word_dict, max_doc_len, stemmer),
                                  [X_train, X_test])

    with h5py.File('train_test.h5', 'w') as hf:
        map(lambda n, d: hf.create_dataset(n, data=d),
                ['X_train', 'X_test', 'y_train', 'y_test', 'word_dict'],
                [X_train_enc, X_test_enc, y_train, y_test, json.dumps(word_dict)])


def load_train_test():
    with h5py.File('train_test.h5', 'r') as hf:
        return map(lambda ds: np.array(hf.get(ds)),
                    ['X_train', 'X_test', 'y_train', 'y_test', 'word_dict'])


def make_lstm_model(embedding_vector_len=32, lstm_output_dim=100):
    model = Sequential(name='lstm')
    model.add(Embedding(words_in_model + 2, embedding_vector_len,
                        input_length=max_doc_length))
    model.add(LSTM(lstm_output_dim))
    model.add(Dense(1, activation='sigmoid'))
    return model


def make_conv_model(embedding_dims=50, nb_filter=250, filter_length=3, hidden_dims=250):
    model = Sequential(name='conv')

    # we start off with an efficient embedding layer which maps our vocab indices into embedding_dims dimensions
    model.add(Embedding(words_in_model + 2, embedding_dims, input_length=max_doc_length, dropout=0.2))

    # we add a Convolution1D, which will learn nb_filter word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))

    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def make_glove_embeddings_index():
    embeddings_index = {}
    with open('glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def train_glove(embedding_dims=100):
    X_train, X_test, y_train, y_test, word_dict_json = load_train_test()
    from StringIO import StringIO
    word_dict = json.load(StringIO(word_dict_json))

    embedding_matrix = np.zeros((len(word_dict) + 2, embedding_dims))
    embeddings_index = make_glove_embeddings_index()

    for word, i in word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_dict) + 2,
                                embedding_dims,
                                weights=[embedding_matrix],
                                input_length=max_doc_length,
                                trainable=False)

    sequence_input = Input(shape=(max_doc_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # happy learning!
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              nb_epoch=2, batch_size=128)


def train(model, batch_size=64, nb_epoch=3):
    X_train, X_test, y_train, y_test, _ = load_train_test()

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size,
              validation_data=(X_test, y_test), nb_epoch=nb_epoch)

    model.save_weights(model.name + "-model.h5")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'feat':
        preprocess_and_save(max_doc_len=max_doc_length, words_in_model=words_in_model)
    elif cmd == 'train-lstm':
        train(make_lstm_model())
    elif cmd == 'train-conv':
        train(make_conv_model())
    elif cmd == 'train-glove':
        train_glove()
    else:
        raise Exception('unknown command ' + cmd)
