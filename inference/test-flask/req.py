import requests

# file = open('/home/filipkr/Documents/xjob/vids/real/Videos/SHIELD/shield-SLS/09SLS1R.mp4', 'rb')

# print(file)


file_name = '/home/filipkr/Documents/xjob/app-mm/03SLS1R_MUSSE.mts'
file_name = '/home/filipkr/Downloads/03SLS1R_MUSSE.mp4'

leg = 'R' if 'R' in file_name.split('/')[-1] else 'L'

# assert False
# files = {'file': open('bulk_test2.mov', 'rb')}
# file = {'file': open('/home/filipkr/Documents/xjob/vids/real/Videos/SHIELD/shield-SLS/09SLS1R.mp4', 'rb')}
# file = {'file': open('/home/filipkr/Documents/xjob/app-mm/03SLS1R_MUSSE.mts', 'rb')}
# file = {'file': open('/home/filipkr/Documents/xjob/vids/real/Videos/Hip-pain/hipp-SLS/10SLS1R.mp4', 'rb')}
# file = {'file': open('/home/filipkr/Documents/xjob/vids/real/Videos/SHIELD/shield-SLS/09SLS1R.mp4', 'rb')}
file = {'file': open(file_name, 'rb')}
# file['file'].close()
# url = 'http://0.0.0.0:5000/'
url_base = 'https://poe-analysis.herokuapp.com/'
url_base = 'http://0.0.0.0:5000/'
url_end = 'get_user'
# url_end = 'get_latest'
# url_end = 'create_user'
url_end = 'upload'
# url_end = 'get_video'
# url_end = 'get_result'
# url_end = 'get_all'
# url_end = 'get_repetition_result'
# url_end = 'delete_user'
url = url_base + url_end
# url = 'http://0.0.0.0:5000/create_user'
# url = "https://poe-analysis.herokuapp.com/upload"

# response = requests.post(url, files={"form_field_name": file})

# DATA TO UPLOAD TO SERVER
id = 'lol3'
# id = 'lol'
id = '940203'
# id = '1994-06-28'
data = {'id': id, 'frames': [1,4,20,600], 'leg': leg, 'weight': '75',
        'length': 185, 'attempt': 1}

data = {'id': id, 'leg': leg, 'attempt': 1, 'frames': [1,2,3]}
# data = {'id': id, 'frames': [1,4,20,600], 'leg': leg, 'weight': '75',
#         'length': 185, 'attempt': 1, 'file': open('/home/filipkr/Documents/xjob/vids/real/Videos/SHIELD/shield-SLS/09SLS1R.mp4', 'rb')}


# POST or GET requests
# response = requests.delete(f'{url}/{id}')
# response = requests.get(url, params=data)
# response = requests.get(url, data=data)
response = requests.post(url, data=data, files=file)
# response = requests.post(url, data=data, files=file, headers={'Content-Type': 'application/octet-stream'})
# response = requests.post(url, data=data)
print(response)
print(response.content)
print(response.request)
# print(response.request.keys)
# print(response.request.body)
print(response.request.path_url)
if response.ok:
    print("Upload completed successfully!")
    # print(type(response.content))
    # print(response.content)
    # resp = str(response.content)
    # print(resp)
    # print(type(resp))
    # print(resp == b'Attempt not in database')
    # print(response.content == b'Attempt not in database')
    # msgs = [b'Attempt not in database', b'lol']
    # print(response.content in msgs)
    # with open(response.headers['Content-Disposition'].split('=')[-1], 'wb') as f:
    #     f.write(response.content)

    print(response.encoding)
    print(response.headers)
    # print(response.headers['Content-Disposition'])
    # print()
    # print(response.apparent_encoding)

    # print(response.file)
else:
    print("Something went wrong!")

# print(resp.text)
