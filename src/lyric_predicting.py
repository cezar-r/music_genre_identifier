import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import nltk
import pickle
from lyric_scraping import get_lyrics
from lyric_cleaning import clean, run
from sklearn.utils import shuffle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify.scikitlearn import SklearnClassifier
from custom_model import NaiveBayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import train_test_split
from statistics import mode
from sklearn.metrics import accuracy_score
plt.style.use("dark_background")

sid = SentimentIntensityAnalyzer()


def text_to_score(x):
	return sid.polarity_scores(x)['compound']


def remove_dups_from_items(arr):
	lyric_history = []
	correct_arr = []
	for lyric, genre in arr:
		if lyric not in lyric_history:
			lyric_history.append(lyric)
			correct_arr.append((lyric, genre))
	return correct_arr


genre_hash = {'hip_hop' : 'Hip-Hop',
			  'country' : 'Country',
			  'r_b' : 'R&B'
}


def build_og_df(genres, df):
	dfs_arr = []
	smallest = 100000
	for genre in genres:

		dfs_arr.append(df[df['genre'] == genre_hash[genre]])
		if len(df[df['genre'] == genre_hash[genre]]) < smallest:
			smallest = len(df[df['genre'] == genre_hash[genre]])
	print(smallest)
	new_dfs_arr = []
	for dfs in dfs_arr:
		new_dfs_arr.append(dfs.iloc[:smallest, :])

	return shuffle(pd.concat(new_dfs_arr, axis=0))


def plot_genre_means(ax, dfs, genres):
	c = ['red', 'green', 'purple']
	for i, genre in enumerate(genres):
		ax.axhline(dfs['sentiment'][dfs['genre'] == genre_hash[genre]].mean(), color=c[i], label=genre, xmin=0, xmax=1)



def pprint(list_of_tups, show = False):

	if not show:
		show = len(list_of_tups) - 1

	print("\nA.I. Guess:")

	list_of_tups = [(k, v) for k, v in sorted(list_of_tups, key = lambda item: item[1])[::-1]]

	sum_guesses = sum([i[1] for i in list_of_tups])

	i = 0
	for tup in list_of_tups:
		if i == show:
			break
		print(f'{tup[0].title()}: {round(tup[1] / sum_guesses * 100, 2)}%')
		i += 1

	print("\nFinal Guess:")
	print(list_of_tups[0][0].title(), '\n')


def pprint2(list_of_tups, show = False):
	if not show:
		show = len(list_of_tups) - 1

	sorted_list_of_tups = {k: v for k, v in sorted(list_of_tups, key=lambda item: item[1])[::-1]}.items()

	i = 0
	for tup in sorted_list_of_tups:
		if i == show:
			break
		if type(tup[0]) == NaiveBayes:
			print(f"Model MyNaiveBayes() accuracy: {round(tup[1] * 100, 2)}%")
		else:
			print(f"Model {tup[0]} accuracy: {round(tup[1] * 100, 2)}%")



def clean_col(x):
	if type(x) == float:
		return 0
	else:
		return x.strip(f'\r')



genre_groups = {'hip-hop' : ['hip-hop', 'hip hop', 'hiphop', 'rap', 'trap', 'atlanta', 'gangsta rap', 'east coast rap'] ,
				'country' : ['country', 'modern country', 'country rock', 'good country', 'big green tractor'],
				'rock' : ['alternative rock', 'indie rock', 'nu metal', 'emo', 'country rock'],
				'punk' : ['alternative', 'punk', 'indie', 'pop punk', 'psychedelic rock', 'skate punk', 'pop-punk', 'indie pop'],
				'edm' : ['dance', 'electronic', 'future bass', 'eurodance', 'house', 'electronica'] }



def under_max_length(arr, genre, max_songs):
	counter = 0
	for i in arr:
		if i == genre:
			counter += 1
	if counter >= max_songs:
		return False
	return True



def get_top_genres(df, n_genres):
	genre_dict = {}
	for genre in df.genre.values:
		if genre in genre_dict:
			genre_dict[genre] += 1
		else:
			genre_dict[genre] = 1

	sorted_dict = {k: v for k, v in sorted(genre_dict.items(), key=lambda item: item[1])[::-1]}
	new_genre_dict = {}
	for k in sorted_dict:
		for genre in genre_groups:
			if k in genre_groups[genre]:
				if genre in new_genre_dict:
					new_genre_dict[genre] += sorted_dict[k]
				else:
					new_genre_dict[genre] = sorted_dict[k]

	max_songs = min(i[1] for i in list(new_genre_dict.items())) * 1.2

	top_genres = []
	for i in range(n_genres):
		top_genres.append(list(new_genre_dict.items())[i][0])


	new_genre_vals = []
	new_lyric_vals = []
	for i in range(len(df.genre.values)):
		for genre in genre_groups:
			if genre in top_genres:
				if under_max_length(new_genre_vals, genre, max_songs):
					if df.genre.values[i] in genre_groups[genre]:
						new_genre_vals.append(genre)
						new_lyric_vals.append(df.lyrics.values[i])

	new_df = pd.DataFrame(list(zip(new_lyric_vals, new_genre_vals)), columns = ['lyrics', 'genre'])
	return new_df



def get_all_words(train, test):
	all_words = set()
	for _lyric in train:
		lyric = _lyric.split(' ')
		for word in lyric:
			all_words.add(word)
	for _lyric in test:
		lyric = _lyric.split(' ')
		for word in lyric:
			all_words.add(word)
	return list(all_words)



def get_all_words2(train):
	all_words = set()
	for _lyric in train:
		lyric = _lyric.split(' ')
		for word in lyric:
			all_words.add(word)
	return list(all_words)


def get_sets(X_train, X_test, y_train, y_test):
	all_words = get_all_words(X_train, X_test)

	training_set = []
	testing_set = []

	X_train = X_train.tolist()
	X_test = X_test.tolist()
	for i, lyric in enumerate(X_train):
		words = set(lyric.split(' '))
		features = {}
		for word in all_words:
			features[word] = (word in words)
		training_set.append((features, y_train.tolist()[i]))

	for i, lyric in enumerate(X_test):
		words = set(lyric.split(' '))
		features = {}
		for word in all_words:
			features[word] = (word in words)
		testing_set.append((features, y_test.tolist()[i]))

	return training_set, testing_set


def get_sets2(X, y):
	all_words = get_all_words2(X)

	training_set = []

	X_train = X.tolist()
	for i, lyric in enumerate(X_train):
		words = set(lyric.split(' '))
		features = {}
		for word in all_words:
			features[word] = (word in words)
		training_set.append((features, y.tolist()[i]))


	return training_set



def run_all_models(X_train, X_test, y_train, y_test, df):
	scores = {}
	models = [NaiveBayes, MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC]

	training_set, testing_set = get_sets(X_train, X_test, y_train, y_test)
	for model in models:
		model = model()
		if type(model) == NaiveBayes:
			model.fit(df)
			y_hat = model.predict(X_test)
			score = model.score(y_test, y_hat, metrics='accuracy')
		else:
			classifier = SklearnClassifier(model)
			classifier.train(training_set)
			score = nltk.classify.accuracy(classifier, testing_set)
		scores[model] = score
	return scores



def test_all_models(X, y, model):

		scores = {}
		models = [model, MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC]

		whole_set = get_sets2(X, y)
		train = whole_set[:len(whole_set)*.8]
		test = whole_set[len(whole_set)*.8:]
		for model in models:
			model = model()
			if type(model) == NaiveBayes:
				y_hat = model.predict(X)
				score = model.score(y, y_hat, metrics='accuracy')
			else:
				classifier = SklearnClassifier(model)
				classifier.train(train)
				score = nltk.classify.accuracy(classifier, test)
			scores[model] = score
		return scores



def run_input(model):

		print('\n')
		song_name = input("Enter song name:\n")
		artist_name = input("Enter artist name:\n")
		print(' ')
		print(f'Listening to "{song_name.title()}" by {artist_name.title()}')
		lyric = clean(get_lyrics(artist_name, song_name))
		scores = model.predict(lyric, full_lyric = False)
		pprint(scores, show = 3)
		go_again = input('Go again? [Y][N]\n')
		if go_again.lower() == 'y':
			run_input(model)
		else:
			return



def main(run_models = False, run_testing = False):
	if run_models:
		df = pd.read_csv("../lyrics/clean_test_12.txt", delimiter = '|', lineterminator='\n')

		df[f'genre\r'] = df.loc[:, f'genre\r'].apply(clean_col)
		df.columns = ['arist_name', 'song_name', 'lyrics', 'genre']

		top_genres_df = get_top_genres(df, 5)
		# use voting system to predict outcome

		df = shuffle(top_genres_df)
		print(np.unique(df.genre.values))
		X, y = df.lyrics.values, df.genre.values
		X_train, X_test, y_train, y_test = train_test_split(X, y)

		all_scores = list(run_all_models(X_train, X_test, y_train, y_test, df).items())
		pprint2(all_scores)

		model = all_scores[0][0]

		save_model = open("model.pickle", "wb")
		pickle.dump(model, save_model)
		save_model.close()
		print('Saved pickle')
		# save to pickle file
	if run_testing:
		# open test_file
		'''
		model_file = open("model.pickle", 'rb')
		model = pickle.load(model_file)
		model_file.close()

		df[f'genre\r'] = df.loc[:, f'genre\r'].apply(clean_col)
		df.columns = ['arist_name', 'song_name', 'lyrics', 'genre']
		top_genres_df = get_top_genres(df, 5)

		df = shuffle(top_genres_df)
		print(np.unique(df.genre.values))
		X, y = df.lyrics.values, df.genre.values

		
		top_genres_df = get_top_genres(df, 5)
		# use voting system to predict outcome
		all_scores = list(test_all_models(X, y, model))


		'''


	else:
		model_file = open("model.pickle", 'rb')
		model = pickle.load(model_file)
		model_file.close()

		# open from pickle file

	run_input(model)
	return


if __name__ == '__main__':
	main(run_models = True)

