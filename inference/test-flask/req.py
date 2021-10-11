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
url_base = 'https://poe-analysis.herokuapp.com/'
url_base = 'http://0.0.0.0:5000/'
url_end = 'get_user'
# url_end = 'get_latest'
# url_end = 'create_user'
# url_end = 'get_all'
# url_end = 'get_repetition_result'
# url_end = 'get_video'
url = url_base + url_end
# url = 'http://0.0.0.0:5000/create_user'
# url = "https://poe-analysis.herokuapp.com/upload"

# response = requests.post(url, files={"form_field_name": file})

# DATA TO UPLOAD TO SERVER
id = '910203'
id = '1994-06-28'
data = {'id': id, 'frames': [1,4,20,600], 'leg': leg, 'weight': '75',
        'length': '185', 'attempt': 1}



# POST or GET requests
# response = requests.post(url, data=data, files=file)
response = requests.get(url, data=data)
print(response)
if response.ok:
    print("Upload completed successfully!")
    print(type(response.content))
    print(response.content)
    # with open(response.headers['Content-Disposition'].split('=')[-1], 'wb') as f:
    #     f.write(response.content)

    # print(response.encoding)
    # print()
    # print(response.apparent_encoding)

    # print(response.file)
else:
    print("Something went wrong!")

# print(resp.text)
