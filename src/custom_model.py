import pandas as pd 
import numpy as np

stop_words = 'a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your'.split(',')

class NaiveBayes:

	def _get_usage(self, d, w):
		total = 0
		for _d in d:
			if w in d[_d]:
				total += d[_d][w]
		return total


	def fit(self, df):
		'''
		meta_dict = {}
		X, y = df['lyrics'], df['genre']

		for genre in y:
			genre_dict = {}
			genre_lyrics = 
		'''

		# creates TF-IDF table

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


	def predict(self, lyric):
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