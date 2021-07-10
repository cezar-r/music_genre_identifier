def clean(string):
	bad_chars = ['</i>', '<i>', '[', ']', '(', ')', 'INTRO:', 'Chorus:', 'CHORUS:', 'Verse:', 'Bridge:', 'Hook:', 'HOOK:']
	for char in bad_chars:
		string = string.replace(char, "")
	return string



def write_clean_file(file):
	with open('clean.txt', 'a') as f:
		f.write('artist|song_name|lyrics|genre\n')
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
			with open('clean.txt', 'a') as f:
				try:
					f.write(artist_name + '|' + song_name + '|' + unfinished_lyrics + '|' + genre + '\n')
				except:
					pass

		elif len(line.split('|')) == 1:

			_lyric = line.strip('\n')
			_lyric = clean(_lyric)

			unfinished_lyrics += _lyric

		else:
			artist_name = line.split('|')[0]
			song_name = line.split('|')[1]
			lyrics = line.split('|')[2]
			lyrics = clean(lyrics)
			genre = line.split('|')[3]


			with open('clean.txt', 'a') as f:
				try:
					f.write(artist_name + '|' + song_name + '|' + lyrics + '|' + genre + '\n')
				except:
					pass


if __name__ == '__main__':
	file = open("hip_hop_data2.txt", "r").readlines()
	write_clean_file(file)
	file = open("edm_data2.txt", "r").readlines()
	write_clean_file(file)


# df = pd.read_csv("clean.txt", delimiter = '|')

# print(df.info())