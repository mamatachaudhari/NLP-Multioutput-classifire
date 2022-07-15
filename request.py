
import requests

url = 'http://localhost:5000/predict'
r = requests.post(url,json={"BP Name": "KARTHIKEYA INFRA - NAGOLE"}) 

print(r.json())

    