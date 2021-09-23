import requests

file = open('/home/filipkr/Documents/xjob/vids/real/Videos/SHIELD/shield-SLS/09SLS1R.mp4', 'rb')

# print(file)


file_name = '/home/filipkr/Documents/xjob/app-mm/03SLS1R_MUSSE.mts'

leg = 'R' if 'R' in file_name.split('/')[-1] else 'L'

# assert False
# files = {'file': open('bulk_test2.mov', 'rb')}
file = {'file': open('/home/filipkr/Documents/xjob/vids/real/Videos/SHIELD/shield-SLS/09SLS1R.mp4', 'rb')}
file = {'file': open('/home/filipkr/Documents/xjob/app-mm/03SLS1R_MUSSE.mts', 'rb')}
# file = {'file': open('/home/filipkr/Documents/xjob/vids/real/Videos/Hip-pain/hipp-SLS/10SLS1R.mp4', 'rb')}
# url = 'http://0.0.0.0:5000/'
url = 'http://0.0.0.0:5000/upload'
url = 'http://0.0.0.0:5000/get_latest'
# url = 'http://0.0.0.0:5000/create_user'
# url = "https://poe-analysis.herokuapp.com/upload"

# response = requests.post(url, files={"form_field_name": file})
data = {'id': '950203', 'frames': [1,4,20,600], 'leg': leg, 'weight': '75',
        'length': '185'}
header = {'Content-type': 'application/json', 'Accept': 'text/plain'}
# response  = requests.post(url, data=data)
# response = requests.post(url, files = {'file': file})
# response = requests.post(url, data=data, files=file)#, headers=header)
response = requests.get(url, data=data)
# print(response.json())
# resp = requests.post("http://127.0.0.1:5000/predict")
#resp = requests.post("https://poe-analysis.herokuapp.com/predict")
if response.ok:
    print("Upload completed successfully!")
    print(response.text)
else:
    print("Something went wrong!")

# print(resp.text)
