import json
import pickle
import torch

from model.Transformer import Transformer

from flask import Flask, request, jsonify

app = Flask(__name__)

# Загружаем конфиг
with open("config.json", "r") as f:
    config = json.load(f)

model_path = f"{config["model_path"]}"

# Определяем устройство (CPU)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

with open("word_field.pkl", "rb") as f:
    word_field = pickle.load(f)

# Создаём объект модели с параметрами из конфига
model = Transformer(
    vocab=word_field.vocab,
    d_model=config["d_model"],
    device=DEVICE,
    embed=config["embed"],
    blocks_count=config["blocks_count"]
).to(DEVICE)

# Загружаем веса
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print("Модель загружена!")

@app.route("/summary", methods=["POST"])
def summary():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Empty input"}), 400

    # Проверяем, является ли вход тензором
    if not isinstance(text, torch.Tensor):
        # Токенизируем текст и превращаем в индексы
        tokens = word_field.preprocess(text)  # Токенизация
        numericalized = [word_field.vocab.stoi[token] for token in tokens]  # Преобразование в индексы
        source_tensor = torch.tensor(numericalized, dtype=torch.long, device=DEVICE).unsqueeze(0)  # Добавляем batch dim

    else:
        source_tensor = text

    # Генерация суммаризации
    gen_tokens = model.generate_only_summary(source_tensor)

    # Декодируем индексы в текст
    summary = " ".join([word_field.vocab.itos[idx] for idx in gen_tokens.squeeze(0).tolist()])

    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
