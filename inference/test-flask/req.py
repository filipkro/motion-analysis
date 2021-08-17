import requests

# resp = requests.post("http://127.0.0.1:5000/predict")
resp = requests.post("https://poe-analysis.herokuapp.com/predict")

print(resp.text)
