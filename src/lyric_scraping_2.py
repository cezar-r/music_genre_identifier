import requests
import json
from lyric_cleaning import clean, write_clean_file
from lyric_scraping import get_lyrics

last_fm_api_key = 'e34c6de6772e623c5f8fac80e4752db4'
base_url = 'http://ws.audioscrobbler.com'
scraper_url = 'http://api.scraperapi.com?api_key=800a43e7386a5a4f5801c8762c3a4aab&url='

pnk_artists = ['my chemical romance', 'saves the day', 'the get up kids', 'the bouncing souls', 'nofx', 'the offspring', 'alkaline trio', 'paramore', 'jimmy eat world', 'buzzcocks', 'green day']
rck_artists = ['spoon', 'john mayer', 'the strokes', 'linkin park', 'u2', 'the white stripes', 'radiohead', 'mgmt', 'brian eno', 'cat power', 'nickelback', 'metallica', 'the walkmen']
rnb_artists = [ 'bow wow', 'the weeknd', 'chris brown' ]
edm_artists = ['the chainsmokers', 'calvin harris', 'steve aoki', 'zedd', 'diplo', 'jauz', 'cheat codes', 'dillon francis']
ctr_artists = ['Jon Pardi', 'Chris Young', 'Garth Brooks', 'Jason Isbell']


# test sets of each genre (5 artists per)
test_pnk_artists = ['Boston Manor', 'The Chats', 'X', 'All time low', 'the boom town rats']
test_rck_artists = ['Fleetwood Mac', 'Queen', 'The Beatles', 'Elton John', 'Pink Floyd']
test_rap_artists = ['Lil Baby', 'Travis Scott', 'Dababy', 'Roddy Ricch', 'Joyner Lucas']
test_edm_artists = ['Flume', 'Diplo', 'Galantis']
test_ctr_artists = ['Lady A', 'Toby Keith', 'Miranda Lambert', 'Rascal Flatts', 'Lee Brice']

artists = test_edm_artists + test_ctr_artists


def clean_song_name(name):
	name = name.split(' ')
	correct_name = ''
	for word in name:
		if word == '':
			continue
		elif word[0] == '#':
			word = word[1:]
			correct_name += word
		elif word[0] != '(':
			correct_name += word + ' '
		else:
			break
	return correct_name[:-1]



def get_songs(artist):
	songs = []
	response = requests.get(base_url + f'/2.0/?method=artist.gettoptracks&artist={artist}&api_key={last_fm_api_key}&format=json')
	json_text = json.loads(response.text)
	tracks = json_text['toptracks']
	track = tracks['track']
	for json_obj in track:
		song_name = json_obj['name']
		songs.append(song_name)
	return songs



def main():
	for artist_name in artists:

		songs = get_songs(artist_name)

		for i, song_name in enumerate(songs):
			print(f'{i+1}/{len(songs)}')
			print(song_name, ' -> ', end='')
			song_name = clean_song_name(song_name)
			print(song_name)

			lyrics = clean(get_lyrics(artist_name, song_name))

			if lyrics.startswith("Exception:"):
				print('rip')
				continue
				# skip this song

			response = requests.get(base_url + f'/2.0/?method=track.getInfo&api_key={last_fm_api_key}&artist={artist_name}&track={song_name}&format=json')
			json_text = json.loads(response.text)
			# print(json.dumps(json_text, indent=4, sort_keys=True))

			if 'error' == list(json_text.keys())[0]:
				print("couldn't find song")
				print(json_text)
				continue
				# next song

			track = json_text['track']
			toptags = track['toptags']
			if toptags == '':
				print('Empty toptags, no genre')
				continue
			tags = toptags['tag']
			for tag in tags:
				if type(tag) == str:
					print('tag was a string, skipping')
					continue 

				genre = tag['name']
				print(song_name, genre)
				if genre == artist_name:
					continue
				else:
					with open(f'../lyrics/new_test_data.txt', 'a') as f:
						print('writing')
						try:
							f.write(artist_name + '|' + song_name + '|' + lyrics + '|' + genre + '\n')
						except:
							pass



def _write_clean(file, outfile, run = False):
	if run:
		file = open(f"../lyrics/{file}.txt", "r").readlines()
		write_clean_file(file, outfile)



_write_clean("clean_test_lyric_data", "clean_test_lyric_data2", run = True)
# main()
# {'error': 6, 'message': 'Track not found'}



