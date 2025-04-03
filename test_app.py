import requests

url = "http://127.0.0.1:5000/summary"
data = {"text": "Это пример новости для суммаризации."}

response = requests.post(url, json=data)
print(response.json())  # Выведет ответ от сервера
