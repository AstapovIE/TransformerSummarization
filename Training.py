from model.Masker import *

import math
from tqdm.auto import tqdm
from rouge import Rouge
import wandb

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

tqdm.get_lock().locks = []


class NoamOpt(object):
    def __init__(self, model, factor=2, warmup=4000, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model.d_model
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def do_epoch(model, criterion, data_iter, optimizer=None, name=None, device="cpu", use_wandb=True):
    name = name or ''
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0
    batches_count = len(data_iter)

    # Список для ROUGE-метрик (если валидация)
    rouge_2_scores = []
    rouge = Rouge()

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count, dynamic_ncols=True, leave=True, desc=name) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch, device)
                # logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])
                logits = model.forward(source_inputs, target_inputs[:, :], source_mask, target_mask[:, :, :])

                logits = logits.contiguous().view(-1, logits.shape[-1])
                # target = target_inputs[:, 1:].contiguous().view(-1)
                target = target_inputs[:, :].contiguous().view(-1)
                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                perplexity = math.exp(loss.item())
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "PPX": f"{perplexity:.2f}"})

                epoch_loss += loss.item()

                # Логируем каждые 500 батчей
                if i % 500 == 0 and use_wandb:
                    wandb.log({"Batch Loss": loss.item()})

                # Для валидации считаем ROUGE на 5 примерах (только на одном батче)
                if not is_train and i == 0 and use_wandb:
                    source_text, target_text, output_text = model.generate_summary(
                        source_inputs[:5], target_inputs[:5], source_mask[:5], target_mask[:5]
                    )

                    # Вычисляем средний ROUGE
                    scores = rouge.get_scores(output_text, target_text, avg=True)
                    rouge_2_scores.append(scores["rouge-2"]["f"])

            final_loss = epoch_loss / batches_count
            final_perplexity = math.exp(final_loss)

            progress_bar.set_postfix({"Final Loss": f"{final_loss:.5f}", "Final PPX": f"{final_perplexity:.2f}"})
            progress_bar.refresh()

    # Если валидация, логируем ROUGE
    if not is_train and use_wandb:
        avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0
        wandb.log({"ROUGE-2": avg_rouge_2})

    return final_loss


def do_epoch_wTF(model, criterion, data_iter, optimizer=None, name=None, device="cpu",
                 use_wandb=True, teacher_forcing_prob=0.6):
    name = name or ''
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0
    batches_count = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count, dynamic_ncols=True, leave=True, desc=name) as progress_bar:
            for i, batch in enumerate(data_iter):
                src, tgt, src_mask = new_convert_batch(batch, device)

                # Инициализируем с <s> токена
                decoder_input = torch.full((src.size(0), 1),
                                           model.vocab[model.sos_token],
                                           device=device)

                logits = []
                for i in range(tgt.size(1) - 1):  # -1 потому что предсказываем на 1 вперёд
                    tgt_mask = subsequent_mask(decoder_input.size(1), device)
                    pad_mask = (decoder_input != model.vocab[model.pad_token]).unsqueeze(1)
                    tgt_mask = tgt_mask & pad_mask

                    output = model(src, decoder_input, src_mask, tgt_mask)
                    logits.append(output[:, -1:])

                    if random.random() < teacher_forcing_prob:
                        next_token = tgt[:, i + 1:i + 2]  # Истинный токен
                    else:
                        next_token = output[:, -1:].argmax(-1)  # Предсказанный токен

                    decoder_input = torch.cat([decoder_input, next_token], dim=1)

                logits = torch.cat(logits, dim=1)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 tgt[:, 1:].reshape(-1))

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                perplexity = math.exp(loss.item())
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}", "PPX": f"{perplexity:.2f}"})

                epoch_loss += loss.item()

                # Логируем каждые 500 батчей
                if i % 500 == 0 and use_wandb:
                    wandb.log({"Batch Loss": loss.item()})

            final_loss = epoch_loss / batches_count

            progress_bar.set_postfix(
                {"Final Loss": f"{epoch_loss / batches_count:.5f}", "Final PPX": f"{math.exp(final_loss):.2f}"})
            progress_bar.refresh()

    return final_loss


def fit(model, criterion, optim, train_iter, epochs_count=1, val_iter=None, device="cpu", use_wandb=True, use_TF=None):
    """use_TF - использовать последовательынй подход с коэф teacher_forcing = use_TF - вероятность подачи
     истинного токена, а не того, который модель выдала на перд.шаге"""
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs_count + 1):
        name_prefix = f'[{epoch} / {epochs_count}] '

        if use_TF:
            train_loss = do_epoch_wTF(model, criterion, train_iter, optim, name_prefix + 'Train:', device, use_wandb,
                                      use_TF)
        else:
            train_loss = do_epoch(model, criterion, train_iter, optim, name_prefix + 'Train:', device, use_wandb)
        train_losses.append(train_loss)

        if val_iter is not None:
            if use_TF:
                val_loss = do_epoch_wTF(model, criterion, val_iter, None, name_prefix + ' Val:', device, use_wandb,
                                        use_TF)
            else:
                val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + ' Val:', device, use_wandb)
            val_losses.append(val_loss)
        else:
            val_loss = None
        if use_wandb:
            wandb.log({
                "Train/Val loss": wandb.plot.line_series(
                    xs=[i for i in range(epoch)],
                    ys=[train_losses, val_losses],
                    keys=["train", "val"],
                    title="Train/Val loss",
                    xname="epoch", )
            })

    return model, train_losses, val_losses


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        """
        vocab_size: размер словаря (количество классов)
        padding_idx: индекс <pad>, который не участвует в вычислении ошибки
        smoothing: коэффициент сглаживания (обычно 0.1)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing  # вероятность истинного класса

    def forward(self, logits, target):
        """
        logits: (batch_size * seq_len, vocab_size) - выход модели
        target: (batch_size * seq_len) - индексы истинных слов
        """
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))  # Распределяем вероятность на все классы
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # Истинному классу даем больше веса
            true_dist[:, self.padding_idx] = 0  # На паддинги вероятность не распределяем

            mask = (target == self.padding_idx)  # Убираем вклад <pad> в лосс
            true_dist[mask] = 0

        return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=-1), dim=-1))
