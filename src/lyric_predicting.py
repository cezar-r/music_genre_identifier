import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lyric_scraping import get_lyrics
from lyric_cleaning import clean
from sklearn.utils import shuffle

testing_size = 60
vocab_size = 30000
embedding_dim = 12
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'


def remove_dups_from_items(arr):
	lyric_history = []
	correct_arr = []
	for lyric, genre in arr:
		if lyric not in lyric_history:
			lyric_history.append(lyric)
			correct_arr.append((lyric, genre))
	return correct_arr



def main():
	df = pd.read_csv("../lyrics/clean.txt", delimiter = '|')

	rap_df = df[df['genre'] == 'Hip-Hop']
	edm_df = df[df['genre'] != 'Hip-Hop']
	rap_df = rap_df.iloc[:len(edm_df), :]

	# print(edm_df.iloc[300, 0:3])
	# return

	dfs = [rap_df, edm_df]
	dfs = pd.concat(dfs, axis=0)
	dfs = shuffle(dfs)


	lyrics = dfs['lyrics'].values
	genres = dfs['genre'].values
	genres = [1 if i == 'Hip-Hop' else 0 for i in genres]
	
	lyric_genre_items = [(j, i) for j, i in zip(lyrics, genres)]
	print(len(lyric_genre_items))
	lyric_genre_items = remove_dups_from_items(lyric_genre_items)
	print(len(lyric_genre_items))

	lyrics = [i[0] for i in lyric_genre_items]
	genres = [i[1] for i in lyric_genre_items]


	# print(len([1 for i in genres if i == 0]))

	X_train = lyrics[testing_size:]
	X_test = lyrics[0:testing_size]

	y_train = genres[testing_size:]
	y_test = genres[0:testing_size]
	# print(len([1 for i in y_test if i == 1]))
	# return

	tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
	tokenizer.fit_on_texts(X_train)

	train_sequences = tokenizer.texts_to_sequences(X_train)
	train_padded = pad_sequences(train_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

	test_sequences = tokenizer.texts_to_sequences(X_test)
	test_padded = pad_sequences(test_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

	train_paddded = np.array(train_padded)
	train_labels = np.array(y_train)
	test_padded = np.array(test_padded)
	test_labels = np.array(y_test)
	print(test_labels)
	print(test_padded)

	model = tf.keras.Sequential([
			tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
			tf.keras.layers.GlobalAveragePooling1D(),
			tf.keras.layers.Dense(24, activation = 'relu'),
			tf.keras.layers.Dense(1, activation = 'sigmoid')])

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	num_epochs = 30
	history = model.fit(train_padded, train_labels, epochs = num_epochs, validation_data = (test_padded, test_labels))
	return model, tokenizer



def predict(model, tokenizer):
	song_name = input("Enter song name:\n")
	artist_name = input("Enter artist name:\n")
	lyrics = clean(get_lyrics(artist_name, song_name))
	print(lyrics)
	sequences = tokenizer.texts_to_sequences(lyrics)
	print(sequences)
	padded = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)
	padded = np.array(padded)
	score = model.predict(padded)
	print(score[0][0])

if __name__ == '__main__':
	model, tokenizer = main()
	predict(model, tokenizer)


'''
Questions

Why is accuracy high when model is running but when I use it, model is very inaccurate
How can I be sure my data is properly cleaned
Is the data I have good enough
Where can I get audio files


//TODO
Clean lyrics even more
	Check if they are long enough
	lower case everything
Review dataframes
Try other models
Use audio files
'''