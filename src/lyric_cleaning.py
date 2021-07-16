def clean(string):
	bad_chars = ['</i>', '<i>', '[', ']', ',', '(', ')', 'INTRO:', 'Chorus:', 'CHORUS:', 'Verse:', 'Bridge:', 'Hook:', 'HOOK:', '—', '–', '�', 'á', 'à', 'â', 'ä', 'ç', 'é', 'è', 'ë', 'í', 'ì', 'ï', 'ì', 'Ò', 'ó', 'ò', 'ö', 'ú', 'ù', 'û', 'ü', 'ß', 'ñ', '…', '’', '‘', '“', '”', '¿', f'\n', "'", '"', ]
	for char in bad_chars:
		string = string.replace(char, "")
		string = ' '.join([i.lower() for i in string.split(' ')])
	return string



def write_clean_file(file, outfile):
	for line in file:
		if len(line.split('|')) == 3:
			artist_name = line.split('|')[0]
			song_name = line.split('|')[1]
			_lyric = line.split('|')[2].strip('\n') + ' '
			_lyric = clean(_lyric)
			unfinished_lyrics = _lyric

		elif len(line.split('|')) == 2:
			_lyric = line.split('|')[0]
			_lyric = clean(_lyric)

			unfinished_lyrics += _lyric
			genre = line.split('|')[1]
			if len(unfinished_lyrics.split(' ')) < 30:
				continue
			else:
				with open(f'../lyrics/{outfile}.txt', 'a') as f:
					try:
						f.write(artist_name + '|' + song_name + '|' + unfinished_lyrics + '|' + genre + '\n')
					except:
						pass

		elif len(line.split('|')) == 1:

			_lyric = line.strip('\n')
			_lyric = clean(_lyric)
			print(_lyric)

			unfinished_lyrics = _lyric

		else:
			artist_name = line.split('|')[0]
			song_name = line.split('|')[1]
			lyrics = line.split('|')[2]
			lyrics = clean(lyrics)
			genre = line.split('|')[3]
		
			if len(lyrics.split(' ')) < 30:
				continue
			else:
				with open(f'../lyrics/{outfile}.txt', 'a') as f:
					try:
						f.write(artist_name + '|' + song_name + '|' + lyrics + '|' + genre + '\n')
					except:
						pass


def run(genres = ['country', 'hip_hop', 'r_b', 'edm']):
	# with open('../lyrics/clean.txt', 'w') as f:
	# 	f.write('artist|song_name|lyrics|genre\n')

	for genre in genres:
		file = open(f"../lyrics/{genre}_data3.txt", "r").readlines()
		write_clean_file(file)
	


# df = pd.read_csv("clean.txt", delimiter = '|')

# print(df.info())