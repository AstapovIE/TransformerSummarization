import torch
from model.Transformer import Transformer

from hometasks_functions import task1, task3
from prepare_data import prepare_data
import json


def compute_metrics():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем данные и модель
    train_iter, test_iter, word_field = prepare_data(DEVICE)

    with open("dvc_pipeline/config.json", "r") as f:
        config = json.load(f)

    # Создаём объект модели с параметрами из конфига
    model = Transformer(
        vocab=word_field.vocab,
        d_model=config["d_model"],
        device=DEVICE,
        embed=config["embed"],
        blocks_count=config["blocks_count"]
    ).to(DEVICE)

    # model = Transformer(vocab=word_field.vocab, d_model=256, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    model.eval()

    task1(model, test_iter, 5, DEVICE, "data/demo_result")
    task3(model, test_iter, 3, DEVICE, "data/attention")

if __name__ == "__main__":
    compute_metrics()