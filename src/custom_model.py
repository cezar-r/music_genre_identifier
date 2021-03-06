import pandas as pd 
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

stop_words = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your'.split(',')

class NaiveBayes:
	"""Custom Naive Bayes model that takes the mean likelihood rather than log sum of likelihoods
    
    Methods
    ----------
    _get_usage(): returns total usage of word across all genres 
    	
    fit(): fits data into tfidf
    	Parameters
    	----------
    	df: pd.DataFrame
    		all data

    predict(): predicts the class based on a lyric
    	Parameters
    	----------
    	lyric: string or array
    		if lyric is a arr, function will recurse over each lyric in arr
    	full_lyric: bool
    		if True, function assumes lyric is of type(arr) will recurse over each lyric in arr

    score(): returns the accuracy of the guesses vs. actual
    	Parameters
    	----------
    	y_test: arr
    		arr of actual classes
    	y_hat: arr
    		arr of predicted classes
    	metrics: str
    		if metrics is accuarcy, function will perform accuracy test
    """

	def _get_usage(self, d, w):
		total = 0
		for _d in d:
			if w in d[_d]:
				total += d[_d][w]
		return total


	def fit(self, df):
		# fits entire vocabulary because the performance of this model is based on mapping words to the class.
		# example: email spam classifier: a model like this would store ALL previous vocabulary to predict future emails. 
		# new vocabulary/words that are not seen before should not affect the model as much.
		meta_dict = {}
		lyrics = df['lyrics']
		uniq_genres = df['genre'].unique()
		for genre in uniq_genres:

			genre_dict = {}
			genre_lyrics = df['lyrics'][df['genre'] == genre]
			for _lyric in genre_lyrics:
				lyric = _lyric.split(' ')
				for _word in lyric:
					word = _word.lower()
					if word != '':
						if word not in stop_words:
							if word in genre_dict:
								genre_dict[word] += 1
							else:
								genre_dict[word] = 1

			meta_dict[genre] = genre_dict

		meta_dict_freq = {}

		all_words = []
		for k in meta_dict:
			for word in meta_dict[k]:
				all_words.append(word)

		for genre in uniq_genres:

			genre_dict_freq = {}

			for word in all_words:
				try:
					total_usage = self._get_usage(meta_dict, word)
					genre_dict_freq[word] = meta_dict[genre][word] / total_usage

				except:
					genre_dict_freq[word] = 0

			meta_dict_freq[genre] = genre_dict_freq
		
		self.tfidf = pd.DataFrame(list(meta_dict_freq.values()), index = list(meta_dict_freq.keys()))
		print(len(self.tfidf.columns))


	def predict(self, lyric, full_lyric = True):
		if not full_lyric:
			lyric = lyric.split(' ')

			probability_dict = {}

			for i, genre in enumerate(list(self.tfidf.index.values)):
				genre_probs = []
				for _word in lyric:
					try:
						word = _word.lower()
						col_idx = list(self.tfidf.columns.values).index(word)
						genre_probs.append(self.tfidf.iloc[i, col_idx])
					except:
						genre_probs.append(0)

				probability_dict[genre] = np.mean(genre_probs)
			return list(probability_dict.items())
		else:
			guesses = []
			for l in lyric:
				guess = self.predict(l, full_lyric = False)
				guess = sorted(guess, key=lambda x: x[1])[::-1]
				guesses.append(guess[0][0])
			return guesses
		

	def score(self, y_test, y_hat, metrics = 'accuracy'):
		if metrics == 'accuracy':
			scores = []
			for i, j in list(zip(y_test, y_hat)):
				if i == j:
					scores.append(1)
				else:
					scores.append(0)
			self._score = sum(scores) / len(scores)
			return self._score


class Council(ClassifierI):
	"""Ensemble model that uses a voting system given a list of classifiers
    
    Methods
    ----------
    classify(): returns prediction of each model 
    	Parameters
    	----------
    	feautes: arr
    		list of data
    """
	def __init__(self, *classifiers):
		self._classifiers = classifiers


	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		return mode(votes)
