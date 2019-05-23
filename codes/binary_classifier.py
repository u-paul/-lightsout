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
import sys, time
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, layers
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

"""

---------------------------------------------------------------- preprocessing --------------------------------------------------------------------------------------

"""


data_set = ['total_3003train', 'total_801val', 'total1103test']

df_train = pd.read_csv('../dataSet/new_data/processed_data/c_data/'+data_set[0]+'.csv')
df_train.drop(df_train.columns[[0]], axis=1, inplace=True)
df_train.dropna(subset=['post'], inplace=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)
train_shape = (df_train.shape[0])
# print("Training shape is ", train_shape)

df_val = pd.read_csv('../dataSet/new_data/processed_data/c_data/'+data_set[1]+'.csv')
df_val.drop(df_val.columns[[0]], axis=1, inplace=True)
df_val.dropna(subset=['post'], inplace=True)
df_val = df_val.sample(frac=1).reset_index(drop=True)
val_shape = (df_val.shape[0])
# print("Val shape is ", val_shape)

first_input = input("Do you want labelled test set?: ")
while (first_input.lower() not in ['y', 'n']):
	print('Please type y or n\n')
	first_input = input("Do you want labelled test set?: ")
if (first_input.lower() == 'y'):
	df_test = pd.read_csv('../dataSet/new_data/processed_data/c_data/'+data_set[2]+'.csv')
	df_test.drop(df_test.columns[[0]], axis=1, inplace=True)
	df_test.dropna(subset=['post'], inplace=True)
	df_test = df_test.sample(frac=1).reset_index(drop=True)
	test_shape = (df_test.shape[0])
	# print("Test shape is ", test_shape)
	df_list = [df_train, df_val, df_test]

else:
	path_to_set = input("Provide the file name. File must be in processed_data/c_data")
	df_test = pd.read_csv('../dataSet/new_data/processed_data/c_data/'+path_to_set+'.csv')
	df_test.dropna(subset=['post'], inplace=True)
	df_list = [df_train, df_val]

x_to_be_printed = df_test['post']
df = pd.concat(df_list)
df = pd.DataFrame(df)
df.dropna(subset=['post'], inplace=True)
total_count = (df.shape[0])

 #incase you want to see the final df
# df.to_csv('../dataSet/new_data/processed_data/c_data/test_111.csv')
# start_time = time.time()
filter = df['post'] != ""
df = df[filter]

## Preprocessing of words

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
exclude = ['my', "isn't", 'until', 'during', 'out', 'from', 'off', 'down', 'while', 'no', 'when', 'not']
for i in exclude:
	if i in STOPWORDS:
		STOPWORDS.discard(i)
print((len(STOPWORDS)))
def clean_text(text):
	try:
		text = re.sub(r"http\S+", "", text)
	except:
		print(text)
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

df['post'] = df['post'].apply(clean_text)
if (first_input.lower() != 'y'):
	df_test['post'] = df_test['post'].apply(clean_text)

# df.to_csv('../dataSet/data4turk/hurricane/final_data/florence.csv', index = False)
print('dpne')

model_input = input("What ML model is to be run?\nType L for linear regression,s for svm, c for CNN, r for RNN: ")



"""
-------------------------------------------------------------- MODELS ---------------------------------------------------------------------------
"""
""" ----------------------------------- Logistic Regression --------------------------------------- """

if (model_input.lower() == 'l'):
	if (first_input.lower() == 'y'):
		posts_train = df['post'][:train_shape+1].values
		y_train = df['label'][:train_shape+1].values 
		upto_test = train_shape + val_shape +1
		posts_test = df['post'][upto_test:].values
		y_test = df['label'][upto_test:].values
		average = []
		labels = ['0', '1']

		vectorizer = CountVectorizer()
		vectorizer.fit(posts_train)
		X_train = vectorizer.transform(posts_train)
		X_test  = vectorizer.transform(posts_test)
		classifier = LogisticRegression()
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		# y_pred = np.argmax(y_pred, axis=1)
		# y_test = np.argmax(y_test, axis=1)

		print(accuracy_score(y_pred, y_test))
		print(classification_report(y_test, y_pred, target_names = labels))

	else:
		average = []
		labels = ['0', '1']
		posts_train = df['post'][:train_shape+1].values
		y_train = df['label'][:train_shape+1].values 
		posts_test = df_test['post'].values
		vectorizer = CountVectorizer()
		vectorizer.fit(posts_train)
		X_train = vectorizer.transform(posts_train)
		X_test  = vectorizer.transform(posts_test)
		classifier = LogisticRegression()
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		annotate = input("Do you want the output csv of annotated data?")
		if (annotate.lower() == 'y'):
			all_posts = []
			predicted_labels = []
			for i in range(len(posts_test)-1):
				all_posts.append(x_to_be_printed[i])
				predicted_labels.append(y_pred[i])
			output_name = input("Provide the output file name. Will be saved under processed_data/c_data")
			out_df = pd.DataFrame({'posts': all_posts,'label': predicted_labels,})
			out_df.to_csv("../dataSet/new_data/processed_data/c_data/"+output_name+".csv")	


###--------------------- End of Linear Regression ----------------------------------------

### ---------------------------------------- SVM ----------------------------------

elif (model_input.lower() == 's'):
	labels = ['0', '1']
	count_vect = CountVectorizer()
	X_train = df['post'][:train_shape+1].values
	y_train = df['label'][:train_shape+1].values
	upto_test = train_shape + val_shape +1
	X_test = df['post'][upto_test:].values
	y_test = df['label'][upto_test:].values
	text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
	text_clf_svm = text_clf_svm.fit(X_train, y_train)
	y_pred = text_clf_svm.predict(X_test)
	print(accuracy_score(y_pred, y_test))
	print(classification_report(y_test, y_pred, target_names = labels))


###--------------------------------- CNN--------------------------------
elif (model_input.lower() in ['c', 'r']):
	MAX_SEQUENCE_LENGTH = 1000
	MAX_NB_WORDS = 20000
	EMBEDDING_DIM = 200

	sentences_train = df['post'][:train_shape+1].values
	y_train = df['label'][:train_shape+1].values
	y_train = to_categorical(np.asarray(y_train))

	val_post = df['post'][train_shape+1:(train_shape+val_shape+1)]
	val_label = df['label'][train_shape+1:(train_shape+val_shape+1)]
	val_label = to_categorical(np.asarray(val_label))
	if (first_input.lower() == 'y'):  
		upto_test = train_shape + val_shape +1
		sentences_test = df['post'][upto_test:].values
		y_test = df['label'][upto_test:].values
		y_test = to_categorical(np.asarray(y_test))
	else:
		sentences_test = df_test['post'].values


	tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
	tokenizer.fit_on_texts(sentences_train)
	tokenizer.fit_on_texts(val_post)

	X_train = tokenizer.texts_to_sequences(sentences_train)
	X_test = tokenizer.texts_to_sequences(sentences_test)
	X_val = tokenizer.texts_to_sequences(val_post)
	
	word_index = tokenizer.word_index

	X_train = pad_sequences(X_train, maxlen = MAX_SEQUENCE_LENGTH)
	X_val = pad_sequences(X_val, maxlen =MAX_SEQUENCE_LENGTH)
	X_test = pad_sequences(X_test, maxlen = MAX_SEQUENCE_LENGTH)

	embeddings_index = {}
	f = open('../dataSet/glove.6B/glove.6B.200d.txt',encoding='utf8')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector

	embedding_layer = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)

	if (model_input.lower() == 'c'):
		l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
		l_pool1 = MaxPooling1D(5)(l_cov1)
		l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
		l_pool2 = MaxPooling1D(5)(l_cov2)
		l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
		l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
		l_flat = Flatten()(l_pool3)
		l_dense = Dense(128, activation='relu')(l_flat)
		preds = Dense(2, activation='sigmoid')(l_dense)
	elif (model_input.lower() == 'r'):
		l_lstm = (LSTM(100))(embedded_sequences)
		preds = Dense(2, activation='sigmoid')(l_lstm)

	model = Model(sequence_input, preds)
	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])
	model.summary()
	cp=ModelCheckpoint('model_cnn.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
	history=model.fit(X_train, y_train, validation_data=(X_val, val_label),epochs=15, batch_size=15,callbacks=[cp])

	y_pred = model.predict(X_test)
	y_pred = np.argmax(y_pred, axis = 1)
	annotate = input("Do you want the output csv of annotated data?")
	if (annotate.lower() == 'y'):
		all_posts = []
		predicted_labels = []
		for i in range(len(X_test)-1):
			all_posts.append(x_to_be_printed[i])
			predicted_labels.append(y_pred[i])
		output_name = input("Provide the output file name. Will be saved under processed_data/c_data")
		out_df = pd.DataFrame({'posts': all_posts,'label': predicted_labels,})
		out_df.to_csv("../dataSet/new_data/processed_data/c_data/"+output_name+".csv")
	else:
		y_test = np.argmax(y_test, axis = 1)
		print(classification_report(y_test, y_pred))







# df1 = pd.read_csv('../dataSet/new_data/processed_data/overall2000test_10.csv')
# df1.drop(df1.columns[[0]], axis=1, inplace=True)
# df1['label'] = 0
# df1.iloc[1989:]["label"] = 1
# df1['label'] = 'unimportant'
# df1.iloc[1991:]["label"] = 'important'
# df1['label'] = df1['label'].map({0: 'unimportant', 1: 'important'})

# df = pd.read_csv('../dataSet/new_data/processed_data/overalltesttrainval.csv')
# df.drop(df.columns[[0]], axis=1, inplace=True)
# filter = df['post'] != ""
# df = df[filter]
# # df = pd.read_csv('../dataSet/new_data/processed_data/overall2000train_' + str(n1-1)+ '.csv')

# REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
# BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))
# exclude = ['my', "isn't", 'until', 'during', 'out', 'from', 'off', 'down', 'while', 'no', 'when', 'not']

# for i in exclude:
# 	if i in STOPWORDS:
# 		STOPWORDS.discard(i)

# first_layer = [250]
# second_layer = [200]
# iterations = [6]

# MAX_SENT_LENGTH = 100
# MAX_SENTS = 15
# MAX_NB_WORDS = 10000
# EMBEDDING_DIM = 200
# labels = []
# texts = []
# reviews = []

# macronum=sorted(set(df['label']))
# macro_to_id = dict((note, number) for number, note in enumerate(macronum))

# def fun(i):
# 	return macro_to_id[i]

# df['label']=df['label'].apply(fun)

# for i in range(df.post.shape[0]):
# 	text = BeautifulSoup(df.post[i])
# 	text=clean_text(str(text.get_text().encode()).lower())
# 	texts.append(text)
# 	sentences = tokenize.sent_tokenize(text)
# 	reviews.append(sentences)




# for i in df['label']:
# 	labels.append(i)


# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(texts)

# data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

# for i, sentences in enumerate(reviews):
# 	for j, sent in enumerate(sentences):
# 		if j< MAX_SENTS:
# 			wordTokens = text_to_word_sequence(sent)
# 			k=0
# 			for _, word in enumerate(wordTokens):
# 				if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
# 					data[i,j,k] = tokenizer.word_index[word]
# 					k=k+1

# word_index = tokenizer.word_index
# print('No. of %s unique tokens.' % len(word_index))

# labels = to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)


# indices = np.arange(data.shape[0])
# 	# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# # nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
# for ii in first_layer:
# 	for jj in second_layer:
# 		for kk in iterations:
# 			x_train = data[:7973]
# 			y_train = labels[:7973]

# 			x_val = data[7973:10071]
# 			y_val = labels[7973:10071]

# 			x_test = data[10071:]
# 			y_test = labels[10071:]


# 			embeddings_index = {}
# 			f = open('../dataSet/glove.6B/glove.6B.200d.txt',encoding='utf8')
# 			for line in f:
# 				values = line.split()
# 				word = values[0]
# 				coefs = np.asarray(values[1:], dtype='float32')
# 				embeddings_index[word] = coefs
# 			f.close()

# 		# `print('Total %s word vectors.' % len(embeddings_index))

# 			embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# 			for word, i in word_index.items():
# 				embedding_vector = embeddings_index.get(word)
# 				if embedding_vector is not None:
# 					embedding_matrix[i] = embedding_vector

# 			embedding_layer = Embedding(len(word_index) + 1,
# 	                            EMBEDDING_DIM,
# 	                            weights=[embedding_matrix],
# 	                            input_length=MAX_SENT_LENGTH,
# 	                            trainable=True)

# 			sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
# 			embedded_sequences = embedding_layer(sentence_input)
# 			l_lstm = Bidirectional(LSTM(ii))(embedded_sequences)
# 			sentEncoder = Model(sentence_input, l_lstm)

# 			review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
# 			review_encoder = TimeDistributed(sentEncoder)(review_input)
# 			l_lstm_sent = Bidirectional(LSTM(jj))(review_encoder)
# 			preds = Dense(len(macronum), activation='softmax')(l_lstm_sent)
# 			model = Model(review_input, preds)
# 			model.compile(loss='categorical_crossentropy',
# 	              optimizer='adam',
# 	              metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])

# 			# model.summary()

# 			cp=ModelCheckpoint('model_han_.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
# 			history=model.fit(x_train, y_train, validation_data=(x_val, y_val),
# 	          epochs= kk, batch_size = 10,callbacks=[cp])

# 			validation_data=(x_test, y_test)
# 			y_pred = model.predict(x_test)
# 			y_pred = np.argmax(y_pred, axis=1)
# 			y_test = np.argmax(y_test, axis=1)
# 			print(classification_report(y_test, y_pred))
# 			print("This result is for {} neurons first layer and {} neurons second layer in {} epochs".format(ii,jj, kk))




# df1['post'] = df1['post'].apply(clean_text)
# filter = df1['post'] != ""
# df1 = df1[filter]
# df1 = df1.sample(frac=1).reset_index(drop=True)

# df = [df, df1]
# df = pd.concat(df)
# df = pd.DataFrame(df)
# print(df.shape)
# df.to_csv('../dataSet/new_data/processed_data/testingtraining1.csv')








"""
-------------------------------------------------------------------------------------- MODELS ------------------------------------------------------------------------------------------------------
"""


"""
---------------------_____ SVM_____-------------------

"""
# from sklearn.linear_model import SGDClassifier

# sgd = Pipeline([('vect', CountVectorizer()),
#                 ('tfidf', TfidfTransformer()),
#                 ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, tol=None)),
#                ])

# X_train, X_test, y_train, y_test = train_posts, test_posts, train_y, test_y
# sgd.fit(X_train, y_train)
# y_pred = sgd.predict(X_test)
# # print(set(y_pred))
# # print("------SVM-------")
# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred, target_names = labels))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
# print(metrics.auc(fpr, tpr))
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.plot(fpr, tpr, marker='.')
# plt.show()

"""
---------------------_____ CNN_____-------------------

"""
# MAX_SEQUENCE_LENGTH = 1000
# MAX_NB_WORDS = 15000
# EMBEDDING_DIM = 100

# macronum = sorted(set(df['label']))
# print(macronum)
# macro_to_id = dict((note,number) for number, note in enumerate(macronum))

# def fun(i):
# 	return macro_to_id[i]
# df['label'] = df['label'].apply(fun)

# posts = []
# labels = []

# posts1 = []
# labels1 = []

# for idx in range(df.post.shape[0]):
# 	post = BeautifulSoup(df.post[idx], "lxml")
# 	posts.append(clean_text(str(post.get_text().encode())))

# for idx in df['label']:
# 	labels.append(idx)


# for idx in range(df1.post.shape[0]):
# 	post = BeautifulSoup(df1.post[idx], "lxml")
# 	posts1.append(clean_text(str(post.get_text().encode())))

# for idx in df1['label']:
# 	labels1.append(idx)

# tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
# tokenizer.fit_on_texts(posts)
# sequences = tokenizer.texts_to_sequences(posts)

# tokenizer1 = Tokenizer(num_words = MAX_NB_WORDS)
# tokenizer1.fit_on_texts(posts1)
# sequences1= tokenizer1.texts_to_sequences(posts1)


# word_index = tokenizer.word_index
# data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# labels = to_categorical(np.asarray(labels))
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]


# word_index1 = tokenizer1.word_index
# data1 = pad_sequences(sequences1, maxlen=MAX_SEQUENCE_LENGTH)
# labels1 = to_categorical(np.asarray(labels1))
# indices1 = np.arange(data1.shape[0])
# np.random.shuffle(indices1)
# data1 = data1[indices1]
# labels1 = labels1[indices1]

# x_train = data
# y_train = labels
# x_val = data1
# y_val = labels1

# embeddings_index = {}
# f = open('../dataSet/glove.6B/glove.6B.100d.txt', encoding = 'utf8')
# for line in f:
# 	values = line.split()
# 	word = values[0]
# 	coefs = np.asarray(values[1:], dtype = 'float32')
# 	embeddings_index[word] = coefs
# f.close()

# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
# 	embedding_vector = embeddings_index.get(word)
# 	if embedding_vector is not None:
# 		embedding_matrix[i] = embedding_vector
# embedding_layer = Embedding(len(word_index) + 1,
#                  EMBEDDING_DIM,weights=[embedding_matrix],
#                  input_length=MAX_SEQUENCE_LENGTH,trainable=True)

# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)

# l_cov1= Conv1D(64, 4, activation='relu')(embedded_sequences)

# l_pool1 = MaxPooling1D(4)(l_cov1)

# l_cov2 = Conv1D(32, 4, activation='relu')(l_pool1)

# l_pool2 = MaxPooling1D(4)(l_cov2)

# l_cov3 = Conv1D(32, 4, activation='relu')(l_pool2)

# l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling

# l_flat = Flatten()(l_pool3)
# l_dense = Dense(64, activation='relu')(l_flat)
    
# preds = Dense(len(macronum), activation='softmax')(l_dense)

# model = Model(sequence_input, preds)
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['acc', keras_metrics.precision(), keras_metrics.recall()])

# cp = ModelCheckpoint('model_cnn.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
# history=model.fit(x_train, y_train, epochs = 10, batch_size = 2,callbacks=[cp])
# y_pred = model.predict(x_val)
# y_pred = np.argmax(y_pred, axis=1)
# y_val = np.argmax(y_val, axis=1)
# print(classification_report(y_val, y_pred))

# score = model.evaluate(x_val, y_val)
# print(model.metrics_names)
# print(score)

# fig1 = plt.figure()
# plt.plot(history.history['loss'],'r', linewidth = 3.0)
# plt.plot(history.history['val_loss'], 'b', linewidth = 3.0)
# plt.legend(['Training loss', 'Validation loss'], fontsize = 18)
# plt.xlabel('Epochs', fontsize = 16)
# plt.ylabel('Loss', fontsize = 16)
# plt.title('Loss curves :CNN', fontsize = 16)
# fig1.savefig('../results/loss_cnn_128')
# plt.show()




"""
-------------------------------------------------------------------- HAN --------------------------------------------------------------------------------------
"""



