import requests
import sys

data = sys.argv[1]

response = requests.post('http://localhost:2801/extract-entities', json = {"data": sys.argv[1]})
print(response.json())
