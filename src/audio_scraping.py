import requests

genres = ['hip-hop', 'classical', 'country', 'electronic', 'metal']

n_pages = 5


def write_to_file(url, genre, i):
	print('got here')
	local_filename = url.split('/')[-1]
	# NOTE the stream=True parameter
	r = requests.get(url, stream=True)
	with open("{}/{}/{:04d}{}".format("audio", genre, i,local_filename), 'wb') as f:
	    for chunk in r.iter_content(chunk_size=1024): 
	        if chunk: # filter out keep-alive new chunks
	            f.write(chunk)
	            #f.flush() commented by recommendation from J.F.Sebastian


for genre in genres:
	counter = 0
	for i in range(1, n_pages+1):
		if i == 1:
			url = f'https://elements.envato.com/audio/genre-{genre}'
		else:
			url = f'https://elements.envato.com/audio/genre-{genre}/pg-{i}'
		new_url = 'http://api.scraperapi.com?api_key=d053cf9f5526234253f49ea9b76a9e7c&url=' + url 
		response = requests.get(new_url).text
		response_split = response.split("src=")
		for split in response_split:
			try:
				idx = split.index('mp3')
				mp3_url = split[idx-78:idx+3]
				print(mp3_url)
				if mp3_url.startswith('https:'):
					write_to_file(mp3_url, genre, counter)
					counter += 1
			except:
				pass
	break
