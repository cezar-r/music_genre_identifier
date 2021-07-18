import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import nltk
import pickle
import warnings
from lyric_scraping import get_lyrics
from lyric_cleaning import clean
from sklearn.utils import shuffle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify.scikitlearn import SklearnClassifier
from custom_model import NaiveBayes, Council
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statistics import mode
from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
plt.style.use("dark_background")
warnings.filterwarnings("ignore")
sid = SentimentIntensityAnalyzer()

# TODO
'''
prob_classify
'''

def text_to_score(x):
	"""Convert a string of text to number between -1 and 1 representing sentiment
    
    Parameters
    ----------
    x: str
    	string of text to be converted into sentiment value
    """
	return sid.polarity_scores(x)['compound']



def pprint(list_of_tups, show = False):
	"""Prints scores for each model
    
    Parameters
    ----------
    list_of_tips: arr
    	array of tuples(model, score)
    show: bool
    	how many models to be shown
    """

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
	"""Prints score for each model
    
    Parameters
    ----------
	list_of_tips: arr
    	array of tuples(model, score)
    show: bool
    	how many models to be shown

    """
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
	"""Used in .apply method to clean each value in a column - strips \r from str, or returns none if value is float (np.nan)
    
    Parameters
    ----------
    x: str
    	string of text to be converted into sentiment value
    """
	if type(x) == float:
		return 0
	else:
		return x.strip(f'\r')



genre_groups = {'hip-hop' : ['hip-hop', 'hip hop', 'hiphop', 'rap', 'trap', 'atlanta', 'gangsta rap', 'east coast rap'] ,
				'country' : ['country', 'modern country', 'country rock', 'good country', 'big green tractor', 'alt-country', 'neo-traditionalist country', 'new country', 'male country'],
				'rock' : ['alternative rock', 'indie rock', 'nu metal', 'emo', 'country rock', 'rock', 'metal', 'classic rock'],
				'punk' : ['alternative', 'punk', 'indie', 'pop punk', 'psychedelic rock', 'skate punk', 'pop-punk', 'indie pop'],
				'edm' : ['dance', 'electronic', 'future bass', 'eurodance', 'house', 'electronica', 'dubstep'] }



def under_max_length(arr, genre, max_songs):
	"""Checks if label appears in array more than max_songs
    
    Parameters
    ----------
    arr: arr
    	list of values that need to be checked on
    genre: str
    	label
    max_songss: int
    	number we do not want to exceed
    """

	counter = 0
	for i in arr:
		if i == genre:
			counter += 1
	if counter >= max_songs:
		return False
	return True



def get_top_genres(df, n_genres, sample_size = None):
	"""Creates a balanced pandas DataFrame with the 5 genres that appear in the genre_hash
	Dataframe that is returned has equal number of songs per genre
    
    Parameters
    ----------
    df: pd.DataFrame
    	DataFrame that is going to be cleaned
    n_genres: int
    	number of genres we want to keep
    sample_size: int
    	number of songs we want per genre
    """
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
	if not sample_size:
		max_songs = min(i[1] for i in list(new_genre_dict.items())) 
	else:
		max_songs = sample_size
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
	"""Gets all words from train and test data
    
    Parameters
    ----------
    train: np.array
    	all words in the train data 
    test: np.array
    	all words in the test data
    """
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
	"""Gets all words from train data
    
    Parameters
    ----------
    train: np.array
    	all words in the train data 
    """
	all_words = set()
	for _lyric in train:
		lyric = _lyric.split(' ')
		for word in lyric:
			all_words.add(word)
	return list(all_words)



def get_sets(X_train, X_test, y_train, y_test):
	"""Gets set for training and testing data
    
    Parameters
    ----------
    X_train: np.array
    	all words in the train data 
    X_test: np.array
    	all words in the test data
    y_train: np.array
    	all labels in the train data
    y_test: np.array
    	all labels in the test data
    """
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


	# training_set/testing_set object explained below
	return training_set, testing_set



def get_sets2(X, y):
	"""Gets set for training data
    
    Parameters
    ----------
    X: np.array
    	all words from data
    y: np.array
    	all labels from data
    """
	all_words = get_all_words2(X)

	training_set = []

	# X_train = X#.tolist()
	X_train = X.tolist()
	for i, lyric in enumerate(X_train):
		words = set(lyric.split(' '))
		features = {}
		for word in all_words:
			features[word] = (word in words)
		training_set.append((features, y.tolist()[i]))

	'''
	training_set is a list of tuples, the first element is a dict where eah key is a keyword and each value is a Boolean. the second element is the class

	[(
	  {
	   keyword1: True, 
	   keyword2: False,
	   keyword3: True
	  }, 
	  genre1
	 ),
	 (
	  {
	   keyword1: False,
	   keyword2: True,
	   keyword3: False
	  }, 
	  genre2
	 )
	]
	'''
	return training_set



def get_votes(test_set, models = None, save = False):
	"""Gets votes from all models and returns the model itself as well as the score
    
    Parameters
    ----------
    test_set: arr
    	all words in test data 
    models: unpacked list
    	all models being used
    save: bool
    	determines whether or not we save the model
    """

	print('getting council votes')
	if save:
		voting_model = Council(*models)

		save_model = open('../models/model-votes.pickle', 'wb')
		pickle.dump(voting_model, save_model)
		save_model.close()
		# print(f'saved pickle {voting_model}')
	else:
		model_file = open('../models/model-votes.pickle', 'rb')
		voting_model = pickle.load(model_file)
		model_file.close()
	score = nltk.classify.accuracy(voting_model, test_set)
	return voting_model, score



def run_all_models(X_train, X_test, y_train, y_test, df):
	"""Runs all models on training data
    
    Parameters
    ----------
    X_train: np.array
    	all words in the train data 
    X_test: np.array
    	all words in the test data
    y_train: np.array
    	all labels in the train data
    y_test: np.array
    	all labels in the test data
    df: pd.DataFrame
    	DataFrame with all words and labels
    """
	scores = {}
	models = [MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC, RandomForestClassifier]
	ran_models = []

	training_set, testing_set = get_sets(X_train, X_test, y_train, y_test)
	for model in models:
		model = model()
		print(model)
		if type(model) == NaiveBayes:
			model.fit(df)
			y_hat = model.predict(X_test)
			score = model.score(y_test, y_hat, metrics='accuracy')
		else:
			classifier = SklearnClassifier(model)
			classifier.train(training_set)
			score = nltk.classify.accuracy(classifier, testing_set)
			ran_models.append(classifier)

			save_model = open(f'../models/model-{model}.pickle', 'wb')
			pickle.dump(classifier, save_model)
			save_model.close()
			print(f'saved pickle {model}')

		scores[model] = score
	vote_model, vote_score = get_votes(testing_set, models = ran_models, save = True)
	scores[vote_model] = vote_score
	return scores



def test_all_models(X, y, model):
	"""Runs all models on unseen testing data
    
    Parameters
    ----------
    X: np.array
    	all words in data
    y: np.array:
    	all labels in data
    df: pd.DataFrame
    	DataFrame with all words and labels
    """
	scores = {}
	models = [model, MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC, RandomForestClassifier]
	whole_set = get_sets2(X, y)
	for model in models:
		print(model)
		if type(model) == NaiveBayes:
			y_hat = model.predict(X)
			score = model.score(y, y_hat, metrics='accuracy')
		elif type(model) == Council:
			score = nltk.classify.accuracy(model, whole_set)
		else:
			model = model()
			model_file = open(f'../models/model-{model}.pickle', 'rb')
			model = pickle.load(model_file)
			model_file.close()

			score = nltk.classify.accuracy(model, whole_set)
		scores[model] = score
	vote_model, vote_score = get_votes(whole_set)
	scores[vote_model] = vote_score
	return scores



def get_word_set_input(lyrics, n_genres = 5):
	"""Gets lyric_bool dict, genre list for individual input
    
    Parameters
    ----------
    lyrics: string
    	words that are getting mapped as boolens into dictionary
    n_genres: int
		numer of genres being classified against
    """
	df = pd.read_csv("../lyrics/clean_lyric_data.txt", delimiter = '|', lineterminator='\n')
	df.columns = ['arist_name', 'song_name', 'lyrics', 'genre']
	df['genre'] = df.loc[:, 'genre'].apply(clean_col)
	top_genres_df = get_top_genres(df, n_genres)
	df = shuffle(top_genres_df)
	X, y = df.lyrics.values, df.genre.values

	all_words = get_all_words2(X)

	genres = ['hip-hop', 'country', 'edm', 'country', 'punk']
	words = set(lyrics.split(' '))
	features = {}
	for word in all_words:
		features[word] = (word in words)
	word_set_dict = {i : [(features, i)] for i in genres}
	return word_set_dict



def run_input():
	"""Allows user to input song name and artist name via command line
    
    Parameters
    ----------
    """
	print('\n')
	song_name = input("Enter song name:\n").title()
	artist_name = input("Enter artist name:\n").title()
	print(' ')
	print(f'Listening to "{song_name.title()}" by {artist_name.title()}')
	lyric = clean(get_lyrics(artist_name, song_name))
	if len(lyric.split(' ')) < 20:
		print("\nCouldn't find song :(")
		go_again = input('Go again? [Y][N]\n')
		if go_again.lower() == 'y':
			run_input()
		return

	model_file = open('../models/model.pickle', 'rb')
	model = pickle.load(model_file)
	model_file.close()

	word_set_dict = get_word_set_input(lyric)
	for genre in word_set_dict:
		score = nltk.classify.accuracy(model, word_set_dict[genre])
		if score == 1:
			break

	print(f'Prediction: {genre.title()}')
	# # get_votes(lyrics)
	# scores = model.predict(lyric, full_lyric = False)
	# pprint(scores, show = 3)
	go_again = input('Go again? [Y][N]\n')
	if go_again.lower() == 'y':
		run_input()
	return


def plot_sentiment(df):
	"""Plots the sentiment of each genre
    
    Parameters
    ----------
    df: pd.DataFrame
    	all data
    """
	unique_vals = np.unique(df.genre.values)
	sent_dict = {}
	for val in unique_vals:
		lyrics = df['lyrics'][df['genre'] == val]
		for lyric in lyrics.values:
			sentiment = text_to_score(lyric)
			if val in sent_dict:
				sent_dict[val].append(sentiment)
			else:
				sent_dict[val] = [sentiment]

	xy = {k : np.mean(v) for k, v in list(sent_dict.items())}
	x = list(xy.keys())
	y = list(xy.values())
	plt.bar(x, y)
	plt.xlabel("Genre")
	plt.ylabel("Sentiment")
	plt.title("Average Sentiment by Genre")
	plt.savefig('../images/sentiment.png')
	# plt.show()



def pprint3(genre_words_dict):
	"""Prints all keywords in genre_keyword dictionary

	genre_word_dict: dict
		dictionary of genre : [keywords] pairs
    """
	for genre in genre_words_dict:
		print(genre.title())
		for word in genre_words_dict[genre]:
			print(word, ' ',  end="")
		print(' ')



def _get_most_common_words(tfidf):
	"""Prints the 30 most important words by class. Value of 30 is hardcoded
	This function avoids words with a likelihood of 1.0 as these are possibly one-offs and may have no true meaning to the genre
    
    Parameters
    ----------
    tfidf: pd.DataFrame
    	tfidf but transposed
    """
	genre_words_dict = {}
	for genre in tfidf.columns.values:
		genre_words_dict[genre]  = [i for i in list(tfidf.sort_values(genre).index.values)[::-1] if tfidf.loc[i, genre] < 1][:30]
	pprint3(genre_words_dict)



def plot_mnist_embedding(X, y, label_map, title=None):
    """Plot an embedding of the mnist dataset onto a plane.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    X: numpy.array, shape (n, 2)
      A two dimensional array containing the coordinates of the embedding.
      
    y: numpy.array
      The labels of the datapoints.  Should be digits.
      
    title: str
      A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.rcParams["figure.figsize"] = (18, 7)

    for i in range(X.shape[0]):
    	plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.tab10(y[i] / 10.), fontdict={'weight': 'bold', 'size': 12})
	
    plt.ylim([.08, .14])
    plt.xlim([0.0005,0.0015])

    patches = []
    for i in range(len(np.unique(y))):
    	patch = mpatches.Patch(color =  plt.cm.tab10(y[y == i][0] / 10.), label = label_map[str(i)] + '-' + str(i))
    	patches.append(patch)

    plt.legend(handles = patches)
    plt.savefig('../images/figure_of_truth_zoomed_in.png')
    # plt.show()



def get_data(df):
	"""Labels data in DataFrame
    
    Parameters
    ----------
    df: pd.DataFrame
    	all data
    """
    data = list(df.lyrics.values)
    labels = df.genre
    le = LabelEncoder()
    y = le.fit_transform(labels)
    seen = []
    genre_label_map = {}
    for label, _y in list(zip(labels, y)):
    	if label not in seen:
    		genre_label_map[str(_y)] = label
    		seen.append(label)
    return data, y, genre_label_map



def get_most_common_words(df):
	"""Parent function to get most common words, initializes tfidf
    
    Parameters
    ----------
    df: pd.DataFrame
    	all data
    """
	model = NaiveBayes()
	model.fit(df)
	tfidf = model.tfidf.T
	_get_most_common_words(tfidf)



def vectorizer(data):
	"""Creates a tfidf and returns it as an np.array, as well as the feature names (vocabulary)
    
    Parameters
    ----------
    data: arr
    	list of all lyrics
    """
	tfidf = TfidfVectorizer()
	X = tfidf.fit_transform(data).toarray()
	return X, np.array(tfidf.get_feature_names())



def plot_genre_labels(df):
	"""Parent function to plotting genre cluster graph
    
    Parameters
    ----------
    df: pd.DataFrame
    	all data
    """
	data, y, label_map = get_data(df)
	vect, vocab = vectorizer(data)
	ss = StandardScaler()
	X = ss.fit_transform(vect)
	pca = PCA(n_components = 5)
	X_pca = pca.fit_transform(X)
	plot_mnist_embedding(X_pca, y, label_map)



def main(run_models = False, 
		n_genres = 5, 
		test_models = False, 
		sample_size = False, 
		user_input = False, 
		test_perf = False, 
		plot_sent = False,
		most_common_words = False,
		genre_plot = False):
	"""Main function that can serve different purpose depending on parameters passed
    
    Parameters
    ----------
    run_models: bool
    	if True, all models will be refitted and retrained with training data
    n_genres: int
    	number of genres we want to classify 
    test_models: bool
    	if True, all models will be tested on unseen data
    sample_size: int
    	how many songs per genre we want. if false, maximum number of songs possible will be used
    user_input: bool
    	if True, program will predict class of song given by user input from command line
    test_perf: bool
		if True, program will test performance of all models with varying sample size and plot (can take up to 24 hours to run)
	plot_sent: bool
		if True, program will plot sentiment of genres
	most_common_words: bool
		if True, program will print most distinguishable words for each genre 
	genre_plot: bool
		if True, program will plot genre cluster plot
    """

	if plot_sent or most_common_words or genre_plot:
		run_models = True
	if run_models:
		df = pd.read_csv("../lyrics/clean_lyric_data2.txt", delimiter = '|', lineterminator='\n')

		df.columns = ['arist_name', 'song_name', 'lyrics', 'genre']
		df['genre'] = df.loc[:, 'genre'].apply(clean_col)

		if not sample_size:
			top_genres_df = get_top_genres(df, n_genres)
		else:
			top_genres_df = get_top_genres(df, n_genres, sample_size = sample_size)

		df = shuffle(top_genres_df)

		if plot_sent:
			plot_sentiment(df)
			return
		if most_common_words:
			get_most_common_words(df)
			return
		if genre_plot:
			plot_genre_labels(df)
			return

		X, y = df.lyrics.values, df.genre.values
		X_train, X_test, y_train, y_test = train_test_split(X, y)

		all_scores = list(run_all_models(X_train, X_test, y_train, y_test, df).items())

		model = all_scores[-1][0]
		save_model = open("../models/model.pickle", "wb")
		pickle.dump(model, save_model)
		save_model.close()
		print('Saved pickle')
		pprint2(all_scores)

	if test_models:
		df = pd.read_csv("../lyrics/clean_test_lyric_data2.txt", delimiter = '|', lineterminator='\n')

		model_file = open("../models/model.pickle", 'rb')
		model = pickle.load(model_file)
		model_file.close()

		df[f'genre\r'] = df.loc[:, f'genre\r'].apply(clean_col)
		df.columns = ['arist_name', 'song_name', 'lyrics', 'genre']
		top_genres_df = get_top_genres(df, n_genres)
		df = shuffle(top_genres_df)

		X, y = df.lyrics.values, df.genre.values

		all_scores = list(test_all_models(X, y, model).items())

		model = all_scores[-1][0]
		save_model = open("../models/model.pickle", "wb")
		pickle.dump(model, save_model)
		save_model.close()
		print('Saved pickle')
		pprint2(all_scores)

	if test_perf:
		test_performance()

	if user_input:
		run_input()

	return all_scores



def plot_performance(scores_dict):
	"""Plots the performance of each model 
    
    Parameters
    ----------
    scores_dict: dict
    	dictionary of model : (size_of data, score) pairs 
    """
	x = [i[0] for i in list(scores_dict.items())[0][1]]
	colors = ['red', 'blue', 'yellow', 'orange', 'green', 'purple', 'cyan', 'fuchsia']
	i = 0
	for model in scores_dict:
		y = [i[1] for i in scores_dict[model]]
		x = [i[0] for i in scores_dict[model]]
		plt.plot(x, y, label = model, color = colors[i])
		i += 1

	plt.legend(loc = 'upper right')
	plt.show()



def test_performance():
	"""Runs all models over range of data size
    
    Parameters
    ----------
    """
	scores_dict = {}
	for i in range(100, 500, 5):
		scores = main(run_models = True, sample_size = i)
		for model, score  in scores:
			if type(model) == Council:
				model = 'Voting Model'
			if f'{model}' in scores_dict:
				scores_dict[f'{model}'].append((i, score))
			else:
				scores_dict[f'{model}'] = [(i, score)]

	plot_performance(scores_dict)


if __name__ == '__main__':
	main(plot_sent = True)

