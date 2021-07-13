import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import nltk
from lyric_scraping import get_lyrics
from lyric_cleaning import clean, run
from sklearn.utils import shuffle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify.scikitlearn import SklearnClassifier
from custom_model import NaiveBayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode

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

	return list_of_tups


def clean_col(x):
	if type(x) == float:
		return 0
	else:
		return x.strip(f'\r')



genre_groups = {'hip-hop' : ['hip-hop', 'hip hop', 'hiphop', 'rap', 'trap', 'atlanta', 'gangsta rap', 'east coast rap'] ,
				'country' : ['country', 'modern country', 'country rock', 'good country', 'big green tractor'] }



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

	max_songs = min(i[1] for i in list(new_genre_dict.items()))

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


def main():
	df = pd.read_csv("../lyrics/new_data.txt", delimiter = '|', lineterminator='\n')

	df[f'genre\r'] = df.loc[:, f'genre\r'].apply(clean_col)
	df.columns = ['arist_name', 'song_name', 'lyrics', 'genre']

	top_genres_df = get_top_genres(df, 2)

	# get top 10 genres
	# shuffle
	# split into train_test, run it through all models
	# use voting system to predict outcome

	dfs = shuffle(top_genres_df)

	model = NaiveBayes()
	model.fit(dfs)

	print('\n')
	song_name = input("Enter song name:\n")
	artist_name = input("Enter artist name:\n")
	print(' ')
	print(f'Listening to "{song_name.title()}" by {artist_name.title()}')
	lyric = clean(get_lyrics(artist_name, song_name))

	scores = model.predict(lyric)
	# scores = [i[1]/sum(scores) for i in scores]
	new_scores = pprint(scores, show = 3)

	print("\nFinal Guess:")
	print(new_scores[0][0].title(), '\n')

	go_again = input('Go again? [Y][N]\n')
	if go_again.lower() == 'y':
		main()
	else:
		return

	return



if __name__ == '__main__':
	# write_new_clean_file(genres, running = False)
	main()

