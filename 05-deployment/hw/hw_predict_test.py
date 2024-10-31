from pprint import pprint

import requests

url = 'http://localhost:9696/predict'

client = {"job": "student", "duration": 280, "poutcome": "failure"}
requests.post(url, json=client).json()

post_request = requests.post(url=url, json=client)
response = post_request.json()
pprint(response)
