import torch
import wandb  # Подключаем wandb
import json
import pickle



from prepare_data import prepare_data
from model.Transformer import Transformer
from Training import LabelSmoothingLoss, NoamOpt, fit
from hometasks_functions import task1, task3


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print("device:", DEVICE)

train_iter, test_iter, word_field = prepare_data(DEVICE)
with open("word_field.pkl", "wb") as f:
    pickle.dump(word_field, f)

config = dict(
    d_model=768,
    device=str(DEVICE),
    blocks_count=4,
    dropout_rate=0.1,
    embed="bert",
    epochs=60,
    use_wandb=True,
    name="Test60b_final",
    model_path = "bert_60.pth"
)

with open("config_bert_60.json", "w") as f:
    json.dump(config, f, indent=4)
print("Конфигурация модели сохранена в config.json")



model = Transformer(vocab=word_field.vocab, d_model=config["d_model"], device=config["device"], embed=config["embed"], blocks_count=config["blocks_count"]).to(DEVICE)
pad_idx = word_field.vocab.stoi['<pad>']
criterion_LB = LabelSmoothingLoss(vocab_size=len(word_field.vocab), padding_idx=pad_idx, smoothing=0.1).to(DEVICE)
optimizer = NoamOpt(model)

# Инициализация wandb
wandb.init(config=config, project="TransformerSummarization", name=config["name"])
# Подключаем модель к wandb для логирования градиентов и параметров
wandb.watch(model, log="all", log_freq=100)

# Обучение
trained_model, train_loss, val_loss = fit(model, criterion_LB, optimizer, train_iter, epochs_count=config["epochs"], val_iter=test_iter,
                                          device=DEVICE, use_wandb=config["use_wandb"], use_TF=None)
#
# Логируем финальные значения после обучения
wandb.log({"Final Train Loss": train_loss, "Final Val Loss": val_loss})
print("train_loss", train_loss, "val_loss", val_loss)

# Сохранение модели
torch.save(trained_model.state_dict(), config["model_path"])
print(f"Модель сохранена в файл {config["model_path"]}")
#
#Загружаем веса
model.load_state_dict(torch.load(config["model_path"], map_location=DEVICE))
model.eval()
print("Модель загружена!")

# Генерация суммаризаций и анализ attention
task1(model, test_iter, 10, DEVICE, f"data/demo_result_{config["embed"]}_{config["epochs"]}")
task3(model, test_iter, 3, DEVICE, f"data/attention_{config["embed"]}_{config["epochs"]}")

# Завершаем логирование
wandb.finish()




config = dict(
    d_model=256,
    device=str(DEVICE),
    blocks_count=4,
    dropout_rate=0.1,
    embed="None",
    epochs=60,
    use_wandb=True,
    name="Test60N_final",
    model_path = "None_60.pth"
)

with open("config_None_60.json", "w") as f:
    json.dump(config, f, indent=4)
print("Конфигурация модели сохранена в config.json")



model = Transformer(vocab=word_field.vocab, d_model=config["d_model"], device=config["device"], embed=config["embed"], blocks_count=config["blocks_count"]).to(DEVICE)
pad_idx = word_field.vocab.stoi['<pad>']
criterion_LB = LabelSmoothingLoss(vocab_size=len(word_field.vocab), padding_idx=pad_idx, smoothing=0.1).to(DEVICE)
optimizer = NoamOpt(model)

# Инициализация wandb
wandb.init(config=config, project="TransformerSummarization", name=config["name"])
# Подключаем модель к wandb для логирования градиентов и параметров
wandb.watch(model, log="all", log_freq=100)

# Обучение
trained_model, train_loss, val_loss = fit(model, criterion_LB, optimizer, train_iter, epochs_count=config["epochs"], val_iter=test_iter,
                                          device=DEVICE, use_wandb=config["use_wandb"], use_TF=None)
#
# Логируем финальные значения после обучения
wandb.log({"Final Train Loss": train_loss, "Final Val Loss": val_loss})
print("train_loss", train_loss, "val_loss", val_loss)

# Сохранение модели
torch.save(trained_model.state_dict(), config["model_path"])
print(f"Модель сохранена в файл {config["model_path"]}")
#
#Загружаем веса
model.load_state_dict(torch.load(config["model_path"], map_location=DEVICE))
model.eval()
print("Модель загружена!")

# Генерация суммаризаций и анализ attention
task1(model, test_iter, 10, DEVICE, f"data/demo_result_{config["embed"]}_{config["epochs"]}")
task3(model, test_iter, 3, DEVICE, f"data/attention_{config["embed"]}_{config["epochs"]}")

# Завершаем логирование
wandb.finish()


