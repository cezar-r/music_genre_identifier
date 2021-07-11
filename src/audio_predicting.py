from pydub import AudioSegment
AudioSegment.converter = "/usr/local/bin/ffmpeg"

sound = AudioSegment.from_mp3("0000preview.mp3")

# get raw audio data as a bytestring
raw_data = sound.raw_data
# get the frame rate
sample_rate = sound.frame_rate
# get amount of bytes contained in one sample
sample_size = sound.sample_width
# get channels
channels = sound.channels