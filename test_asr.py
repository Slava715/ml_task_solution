from pydub import AudioSegment
import requests
import io
import sys

headers = {'num_norm': 'yes', 'punct': 'yes'}

sound = AudioSegment.from_file(sys.argv[1], format=sys.argv[2])
sound = sound.set_frame_rate(16000)

channels = sound.split_to_mono()
f_wav = io.BytesIO()

for channel in channels:
	channel.export(f_wav, format="raw")
	response = requests.post('http://localhost:2701/asr_file_raw', data = f_wav, headers = headers)
	print(response.json())
