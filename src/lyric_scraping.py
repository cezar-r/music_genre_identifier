import requests
import json
import re 
import urllib.request 
from bs4 import BeautifulSoup


headers = {
    'x-rapidapi-key': "9237b582dbmsh4764c18bdbf6380p1ef6f2jsn6e2ee0f062a7",
    'x-rapidapi-host': "theaudiodb.p.rapidapi.com"
    }


hip_hop_artists = ['Eminem', '50 Cent', 'J. Cole', 'Drake', 'A$AP Ferg', 'Snoop Dogg', 'Rick Ross', 
				   'Lil Wayne', 'A$AP Rocky', 'Nicki Minaj', 'Kanye West', 'Meek Mill', '2 Chainz', 
				   'Mac Miller', 'Childish Gambino', 'Future', 'MF Doom', 'Big Sean', 'Kendrick Lamar', 'Jay-Z',]


country_artists = ['The Band Perry',
				   'Eric Church', 'Jake Owen', 'Luke Bryan', 'Carrie Underwood', 'Trace Adkins',
				   'Darius Rucker', 'Toby Keith', 'Joe Nichols'] 
				   # 'Taylor Swift', 'Blake Shelton', 'Keith Urban', 'Miranda Lambert', 'Jason Aldean', 'Kenny Chesney', 


r_b_artists = ['The Weeknd', 'Ne-Yo', 'Kid Cudi', 'Alicia Keys', 'Beyonce',
			   'Miguel', 'Usher', 'Charlie Wilson', 'Kelly Price', 'Chris Brown',
			   'Jeremih', 'Bruno Mars', 'Estelle', 'Luke James', 'John Legend',
			   'TLC', 'Aaliyah']


edm_artists = ['Daft Punk', 'Skrillex', 'Diplo', 'Tiesto', 'David Guetta', 'Deadmau5',
			   'Zedd', 'Swedish House Mafia', 'Avicii', 'Calvin Harris', 'Duke Dumont',
			   'Disclosure', 'Major Lazer', 'Afrojack', 'Coldplay']


def get_albums(artist_group):
	url = "https://theaudiodb.p.rapidapi.com/searchalbum.php"
	albums_dict = {}
	for artist in artist_group:
		print(artist)
		albums = []
		querystring = {"s":artist}
		response = requests.request("GET", url, headers=headers, params=querystring)
		response = json.loads(response.text)
		list_of_albums = response['album']
		for _album in list_of_albums:
			albums.append(_album['idAlbum'])
		albums_dict[artist] = albums
	return albums_dict



def get_genre_data(albums_dict):
	url = "https://theaudiodb.p.rapidapi.com/track.php"
	genre_data = []
	for artist in albums_dict:
		_artist = artist
		if _artist == 'A$AP Rocky': 
			_artist = 'ASAP Rocky'
		elif _artist == 'A$AP Ferg': 
			_artist = 'ASAP Ferg'
		artist_dict = {'artist' : "".join(_artist.lower().split(" "))}
		_data = {}
		for album in albums_dict[artist]:
			querystring = {"m":album}
			response = requests.request("GET", url, headers=headers, params=querystring)
			response = json.loads(response.text)
			list_of_tracks = response['track']
			for tracks in list_of_tracks:
				if tracks['strGenre'] is not None:
					_data[tracks['strTrack']] = tracks['strGenre']
			artist_dict['data'] = _data
		genre_data.append(artist_dict)
	return genre_data
 

 
def get_lyrics(artist,song_title): 
    artist = artist.lower() 
    song_title = song_title.lower() 
    # remove all except alphanumeric characters from artist and song_title 
    artist = re.sub('[^A-Za-z0-9]+', "", artist) 
    song_title = re.sub('[^A-Za-z0-9]+', "", song_title) 
    if artist.startswith("the"):    # remove starting 'the' from artist e.g. the who -> who 
        artist = artist[3:] 
    url = "http://azlyrics.com/lyrics/"+artist+"/"+song_title+".html"
    print(url)
    new_url = 'http://api.scraperapi.com?api_key=d053cf9f5526234253f49ea9b76a9e7c&url=' + url 
     
    try: 
        content = urllib.request.urlopen(new_url).read() 
        soup = BeautifulSoup(content, 'html.parser') 
        lyrics = str(soup) 
        # lyrics lies between up_partition and down_partition 
        up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->' 
        down_partition = '<!-- MxM banner -->' 
        lyrics = lyrics.split(up_partition)[1] 
        lyrics = lyrics.split(down_partition)[0] 
        lyrics = lyrics.replace('<br/>','').replace('</br>','').replace('</div>','').strip().replace('\n', ' ')
        return lyrics 
    except Exception as e: 
        return "Exception occurred \n" +str(e) 



def clean_song_name(name):
	name = name.split(' ')
	correct_name = ''
	for word in name:
		if word[0] != '(':
			correct_name += word + ' '
		else:
			break
	return correct_name[:-1]



def write_data(genre_data, g_type):
	for artist_dict in genre_data:
		artist_name = artist_dict['artist']
		artist_data = artist_dict['data']
		i = 1
		for song_name in artist_data:
			print(f'{i}/{len(list(artist_data.keys()))}')
			print(song_name)
			print(artist_name)
			genre = artist_data[song_name]
			song_name = clean_song_name(song_name)
			lyrics = get_lyrics(artist_name, song_name)
			if lyrics.startswith("Exception"):
				pass
			else:
				with open(f'{g_type}_data2.txt', 'a') as f:
					try:
						f.write(artist_name + '|' + song_name + '|' + lyrics + '|' + genre + '\n')
					except:
						pass
			i += 1



if __name__ == '__main__':
	albums_dict = get_albums(edm_artists)
	genre_data = get_genre_data(albums_dict)
	write_data(genre_data, "edm")

