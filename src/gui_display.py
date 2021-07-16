import PySimpleGUI as sg
import nltk
import pickle
sg.theme("DarkBlack")

model_file = open('../models/model.pickle', 'rb')
model = pickle.load(model_file)
model_file.close()


class GUI:

	layout = [[sg.Text('Enter song name'), sg.Input(background_color = 'black', text_color = 'white', key = 'i1')],
			[sg.Text('Enter artist name'), sg.Input(background_color = 'black', text_color = 'white', key = 'i2')],
			[sg.Button('Run', button_color = 'fuchsia'), sg.Button('Clear', button_color = 'white')],
			[sg.Text('Prediction:')],
			[sg.Text('              ', key = 'pred', size=(15,1))]]

	window = sg.Window('A.I. Genre Classifier', layout)


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
