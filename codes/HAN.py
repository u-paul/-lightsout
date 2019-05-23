import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import itertools
import re
import keras_metrics
import emoji
from bs4 import BeautifulSoup
import sys
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from sklearn.metrics import classification_report
import os
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers


df = pd.read_csv('../dataSet/main_data/cleanv1.csv', header= 0)



# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()

# def clean_str(string):
#     string = re.sub(r"\\", "", string)
#     string = re.sub(r"\'", "", string)
#     string = re.sub(r"\"", "", string)
#     return string.strip().lower()



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
exclude = ['my', "isn't", 'until', 'during', 'out', 'from', 'off', 'down', 'while', 'no', 'when', 'not']

for i in exclude:
	if i in STOPWORDS:
		STOPWORDS.discard(i)
print((len(STOPWORDS)))

# def clean_text(text):
# 	text = BeautifulSoup(text, "lxml").text # HTML decoding
# 	text = text.lower() # lowercase text
# 	text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
# 	text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
# 	text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ",text).split())
# 	text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
# 	text = emoji.demojize(text)
# 	text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
# 	text = text.replace(":", " ")
# 	text = ' '.join(text.split())
# 	return text

def clean_str(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = BeautifulSoup(text).text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ",text).split())
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = emoji.demojize(text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text = text.replace(":", " ")
    text = ' '.join(text.split())
    return text.strip().lower()




MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2



df['post'] = df['post'].apply(clean_str)


labels = []
texts = []
reviews = []

macronum=sorted(set(df['label']))
macro_to_id = dict((note, number) for number, note in enumerate(macronum))

def fun(i):
    return macro_to_id[i]

df['label']=df['label'].apply(fun)

for i in range(df.post.shape[0]):
    text = BeautifulSoup(df.post[i])
    text=clean_str(str(text.get_text().encode()).lower())
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)


for i in df['label']:
    labels.append(i)


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1

word_index = tokenizer.word_index
print('No. of %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


embeddings_index = {}
f = open('../dataSet/glove.6B/glove.6B.200d.txt')
 # ,encoding='utf8'
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
print('shape of sentence input is', sentence_input, type(sentence_input))
# embedded_sequences = embedding_layer(sentence_input)
# l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
# sentEncoder = Model(sentence_input, l_lstm)

# review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
# review_encoder = TimeDistributed(sentEncoder)(review_input)
# l_lstm_sent = Bidirectional(LSTM(150))(review_encoder)
# preds = Dense(len(macronum), activation='softmax')(l_lstm_sent)
# model = Model(review_input, preds)

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])

# print("Hierachical LSTM")
# model.summary()

# cp=ModelCheckpoint('model_han_.hdf5',monitor='val_acc',verbose=1,save_best_only=True)

# history=model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           epochs=5, batch_size=2,callbacks=[cp])

# validation_data=(x_test, y_test)
# y_pred = model.predict(x_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_val = np.argmax(y_test, axis=1)
# print(classification_report(y_test, y_pred))
# score = model.evaluate(x_val, y_val)
# print(model.metrics_names)
# print(score)


# fig1 = plt.figure()
# plt.plot(history.history['loss'],'r',linewidth=3.0)
# plt.plot(history.history['val_loss'],'b',linewidth=3.0)
# plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Loss',fontsize=16)
# plt.title('Loss Curves :HAN',fontsize=16)
# # fig1.savefig('loss_han.png')
# plt.show()

# fig2=plt.figure()
# plt.plot(history.history['acc'],'r',linewidth=3.0)
# plt.plot(history.history['val_acc'],'b',linewidth=3.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Accuracy',fontsize=16)
# plt.title('Accuracy Curves : HAN',fontsize=16)
# # fig2.savefig('accuracy_han.png')
# plt.show()