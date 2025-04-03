from torchtext.data import Field, Example, Dataset, BucketIterator
import pandas as pd
from tqdm.auto import tqdm

import os
import pickle
import requests
import zipfile
import json


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
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/news.csv"):
        print("Файл news.csv не найден. Начинаем скачивание...")
        url = "https://drive.google.com/uc?export=download&id=1hIVVpBqM6VU4n3ERkKq4tFaH4sKN0Hab"
        zip_path = "data/news.zip"

        # Скачиваем файл
        download_file(url, zip_path)

        # Распаковываем
        unzip_file(zip_path, extract_to="data")

        print("Скачивание и разархивация завершены.")
    else:
        print("[INFO] Файл news.csv уже существует, скачивание не требуется.")


def preprocess_and_cache_data(data, word_field, fields, cache_file="data/examples.pkl"):
    if os.path.exists(cache_file):
        print(f"[INFO] examples для обучения уже есть. Загрузка данных из кэша: {cache_file}")
        with open(cache_file, "rb") as f:
            examples = pickle.load(f)
    else:
        print("Кэш examples для обучения не найден. Подготовка данных...")
        examples = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Preparing data"):
            source_text = word_field.preprocess(row.text)
            target_text = word_field.preprocess(row.title)
            examples.append(Example.fromlist([source_text, target_text], fields))

        with open(cache_file, "wb") as f:
            pickle.dump(examples, f)
        print(f"Данные сохранены в {cache_file}")

    return examples


def prepare_data(device):
    download_data()

    SOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    word_field = Field(tokenize='moses', init_token=SOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    fields = [('source', word_field), ('target', word_field)]

    data = pd.read_csv('data/news.csv', delimiter=',')
    examples = preprocess_and_cache_data(data, word_field, fields)

    # building datasets
    dataset = Dataset(examples, fields)

    import random
    random.seed(42)  # Фиксируем случайность
    train_dataset, test_dataset = dataset.split(split_ratio=0.85)

    word_field.build_vocab(train_dataset, min_freq=7)

    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset), batch_sizes=(16, 32), shuffle=True, device=device, sort=False
    )
    return train_iter, test_iter, word_field


# if __name__ == "__main__":
#     download_data()


