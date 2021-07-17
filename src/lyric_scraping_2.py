import requests
import json
from lyric_cleaning import clean, write_clean_file
from lyric_scraping import get_lyrics

last_fm_api_key = """ENTER LAST FM API KEY HERE"""
base_url = 'http://ws.audioscrobbler.com'
scraper_url = """ENTER SCRAPER URL HERE"""

artists = ["""ENTER CONCATED LIST OF ARTISTS HERE"""]


def clean_song_name(name):
	"""Cleans the name of a song 
	ex: "song_name (feat. artist_name)"" -> "song_name"
    
    Parameters
    ----------
    name: str
    	string of song name that is getting cleaned
    """
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
	"""Gets a list of all songs for a given artist
    
    Parameters
    ----------
    artist: str
    	string of artist we are getting songs for
    """
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
	"""Runs through list of all artists and each song for that artist
	Fetches lyrics and genre for that song and writes out to file
    
    Parameters
    ----------
    """
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
	"""Writes a new clean file from a given file
    
    Parameters
    ----------
    file: str
    	string of file name we are getting lyrics from 
    outfile: str
    	string of file name we are writing lyrics into
    run: bool
    	if True function will run, added as a layer of security so as to not accidentally overwrite file.
    """
	if run:
		file = open(f"../lyrics/{file}.txt", "r").readlines()
		write_clean_file(file, outfile)



if __name__ == '__main__':
	main()




