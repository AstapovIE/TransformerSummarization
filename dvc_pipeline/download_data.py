import os
import requests
import zipfile

def download_file(url, filename):
    print(f"Скачивание {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"Файл {filename} загружен.")


def unzip_file(zip_path, extract_to="."):
    print(f"Распаковка {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Распаковка завершена.")


def download_data():
    if not os.path.exists("../data"):
        os.mkdir("../data")
    if not os.path.exists("../data/news.csv"):
        print("Файл news.csv не найден. Начинаем скачивание...")
        url = "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab"
        zip_path = "../data/news.zip"

        # Скачиваем файл
        download_file(url, zip_path)

        # Распаковываем
        unzip_file(zip_path, extract_to="../data")

        print("Скачивание и разархивация завершены.")
    else:
        print("[INFO] Файл news.csv уже существует, скачивание не требуется.")


if __name__ == "__main__":
    download_data()