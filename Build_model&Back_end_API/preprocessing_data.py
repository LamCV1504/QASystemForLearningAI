import io
import json

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors

from config import get_path, get_path_currying
from config import EMBEDDING_DIM, NUM_WORDS, MAX_LEN, PAD


# Data_Full_train_2C = "D:\\NCKH_HTHD\\Data\\all.xlsx"
Data_Full_train_2C = get_path('Data\\all.xlsx')

# url_word2vec_aspect = "D:\\Resources\\ALL_CHAPTER.model"
resource_path = get_path_currying('Resources\\')
url_word2vec_aspect = resource_path('ALL_CHAPTER.model') 

def load_data_temp():

    train_data = pd.read_excel(Data_Full_train_2C, 'dataset')

    print(train_data.isnull().sum())

    # dic = {'tongquantimkiem':0, 'timkiemmu':1, 'timkiemcothongtin':2, 'timkiemcucbo':3}
    # labels = train_data.label2.apply(lambda x: dic[x])
    dic = {'chuong1':0, 'chuong2':1, 'chuong3':2, 'chuong4':3, 'chuong5': 4}
    labels = train_data.label1.apply(lambda x: dic[x])

    print(labels)

    NUM_CLASS = len(dic)

    text1 = []
    for row in train_data.text:
        text1.append(row)
 
    val_data = train_data.sample(frac = 0.1, random_state=90) 
    test_data =  val_data
    train_data = train_data.drop(val_data.index)

    texts = train_data.text

    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',lower=True)
    tokenizer.fit_on_texts(text1)
    
    tokenizer_json = tokenizer.to_json()
    with io.open(resource_path('word2vec\\vocab.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid = tokenizer.texts_to_sequences(val_data.text)
    sequences_test = tokenizer.texts_to_sequences(test_data.text)

    word_index = tokenizer.word_index
    X_train = pad_sequences(sequences_train, maxlen=MAX_LEN , padding=PAD[0])
    X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1], padding=PAD[0])
    X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1], padding=PAD[0])
    
    y_train = to_categorical(np.asarray(labels[train_data.index]),num_classes=NUM_CLASS)    
    y_val = to_categorical(np.asarray(labels[val_data.index]),num_classes=NUM_CLASS)
    y_test = to_categorical(np.asarray(labels[test_data.index]),num_classes=NUM_CLASS)
  
    word_vectors = KeyedVectors.load(url_word2vec_aspect, mmap='r')
    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

    del (word_vectors)

    from keras.layers import Embedding
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)
 
    return X_train, y_train, X_test, y_test, X_val, y_val, embedding_layer
