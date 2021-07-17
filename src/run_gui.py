from gui_display import GUI
from lyric_scraping import get_lyrics
from lyric_cleaning import clean
from lyric_predicting import get_word_set_input


def run():
	g = GUI(get_lyrics, clean, get_word_set_input)
	return

run()