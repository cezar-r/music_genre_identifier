import PySimpleGUI as sg
import nltk
import pickle
sg.theme("DarkBlack")

model_file = open('../models/model.pickle', 'rb')
model = pickle.load(model_file)
model_file.close()


class GUI:
	"""GUI class that runs PySimpleGUI object
    
    Methods
    ----------
    predict(): predict class of genre based on user input
    	Parameters
    	----------
    	song_name: str
    		string of song name
    	artist_name: str
    		string of artist name
    """
	layout = [
		[sg.Text('Enter song name', 
				font=('Segoe UI Semibold', 20)
			), 
			sg.Input(background_color = 'black', 
				text_color = 'white', 
				key = 'i1', 
				font=('Segoe UI Semibold', 20)
				)
		],
		[sg.Text('Enter artist name', 
				font=('Segoe UI Semibold', 20)
			), 
			sg.Input(background_color = 'black', 
				text_color = 'white', 
				key = 'i2', 
				font=('Segoe UI Semibold', 20)
				)
		],
		[sg.Button('Run', 
				button_color = 'fuchsia', 
				size=(15, 0)
			  ), 
			sg.Button('Clear', 
				button_color = 'white', 
				size=(15, 0)
				 )
		],
		[sg.Text('Prediction:', 
				font=('Segoe UI Semibold', 20)
			)
		],
		[sg.Text('              ', 
				font=('Segoe UI Semibold', 20), 
				key = 'pred', 
				size=(25,1)
			)
		]
	]

	window = sg.Window('A.I. Genre Classifier', layout, size=(500, 300))


	def __init__(self, *funcs):
		self._funcs = funcs

		while True:
			event, values = GUI.window.read()
			if event == sg.WIN_CLOSED:
				break
			if event == 'Run':
				self.predict(values['i1'], values['i2'])
				GUI.window['pred'].update(value = self.genre, text_color = 'fuchsia')
			if event == 'Clear':
				GUI.window['i1'].update(value = '')
				GUI.window['i2'].update(value = '')
				GUI.window['pred'].update(value = '')

		GUI.window.close()


	def predict(self, song_name, artist_name):
		lyric = self._funcs[1](self._funcs[0](artist_name, song_name))
		if len(lyric.split(' ')) < 20:
			self.genre = "Couldn't find song :("
			return 
		word_set_dict = self._funcs[2](lyric)

		for genre in word_set_dict:
			score = nltk.classify.accuracy(model, word_set_dict[genre])
			if score == 1:
				break
		self.genre = genre
