import requests

response = requests.post("http://localhost:5000/predict", files={"file": open('cat.jpeg','rb')})

print(response.json())