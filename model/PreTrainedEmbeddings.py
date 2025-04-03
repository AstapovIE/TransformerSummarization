import fasttext
import fasttext.util
import torch
from torch.xpu import device
from tqdm import tqdm
import gzip
import shutil
import os
import wget


def download_fasttext_emb():
    if not os.path.exists("../data/cc.ru.300.bin"):
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz"
        output_path = "../data/cc.ru.300.bin.gz"

        if not os.path.exists(output_path):
            print("Скачивание FastText embeddings... Происходит через wget из pip, может работать нестабильно :)")
            wget.download(url, output_path)
            print("\nФайл скачан!")
        else:
            print("Файл уже загружен.")

        print("Разархивируем файл...")
        with gzip.open(output_path, "rb") as f_in:
            with open("../data/cc.ru.300.bin", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print("Готово! Файл сохранен как cc.ru.300.bin")
    else:
        print("Файл cc.ru.300.bin уже есть")



def get_bert_emb(vocab, save_path="data/bert_emb.pt"):
    if os.path.exists(save_path):
        print(f"[INFO] Загружаем сохранённые эмбеддинги из {save_path} ...")
        embedding_matrix = torch.load(save_path)
    else:
        print("Скачиваем и загружаем модель BERT...")

        from transformers import AutoModel, AutoTokenizer
        # Загружаем токенизатор и модель
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        bert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

        vocab_size = len(vocab)
        embedding_matrix = torch.zeros((vocab_size, 768), dtype=torch.float32)

        print("Создаём матрицу эмбеддингов...")
        for word, idx in tqdm(vocab.stoi.items(), desc="Заполняем эмбеддинги", unit="слово"):
            inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                # Извлекаем последние скрытые состояния
            hidden_states = outputs.last_hidden_state  # tensor of shape (batch_size, sequence_length, hidden_size)
            # Для каждого слова обычно будет несколько токенов, поэтому делаем усреднение по всем токенам
            embedding_matrix[idx] = hidden_states.mean(dim=1).squeeze() # усреднение по токенам (batch_size, hidden_size)

        print(f"Сохраняем эмбеддинги в {save_path}...")
        torch.save(embedding_matrix, save_path)

    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

    return embedding_layer


def get_fasttext_emb(vocab, emb_size=300, save_path="data/fasttext_emb"):
    path_w_dim = f"{save_path}_{emb_size}.pt"
    if os.path.exists(path_w_dim):
        print(f"[INFO] Загружаем сохранённые эмбеддинги из {path_w_dim} ...")
        embedding_matrix = torch.load(path_w_dim)
    else:
        print("Скачиваем и загружаем модель FastText...")
        download_fasttext_emb()
        # cc.ru.300.bin скорей всего надо будет скачивать вручную из интернета
        fasttext_model = fasttext.load_model('../data/cc.ru.300.bin')
        fasttext.util.reduce_model(fasttext_model, emb_size)

        vocab_size = len(vocab)
        embedding_matrix = torch.zeros((vocab_size, emb_size), dtype=torch.float32)

        print("Создаём матрицу эмбеддингов...")
        for word, idx in tqdm(vocab.stoi.items(), desc="Заполняем эмбеддинги", unit="слово"):
            if word in fasttext_model.words:
                embedding_matrix[idx] = torch.tensor(fasttext_model.get_word_vector(word))
            else:
                embedding_matrix[idx] = torch.randn(emb_size)

        print(f"Сохраняем эмбеддинги в {path_w_dim}...")
        torch.save(embedding_matrix, path_w_dim)

    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

    return embedding_layer




