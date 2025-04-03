import torch

from model.Encoder import Encoder
from model.Decoder import Decoder
from model.basic_layers import *
from model.PreTrainedEmbeddings import get_fasttext_emb, get_bert_emb
from model.Masker import *


class Transformer(nn.Module):
    def __init__(self, vocab, d_model=256, d_ff=1024, blocks_count=4, heads_count=8, dropout_rate=0.1, embed="None",
                 device="cpu", sos_token='<s>', eos_token='</s>', pad_token='<pad>'):
        super(Transformer, self).__init__()
        self.embed = embed
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.device = device

        if embed == "bert" and d_model != 768:
            raise ValueError("If using BERT, d_model must be 768")

        if embed == "fasttext":
            print("[INFO] Используем предобученные FastText эмбеддинги.")
            self.embedding = get_fasttext_emb(vocab, emb_size=d_model)
        elif embed == "bert":
            print("[INFO] Используем bert эмбеддинги.")
            self.embedding = get_bert_emb(vocab)
        else:
            print("[INFO] Используем стандартные эмбеддинги.")
            self.embedding = nn.Embedding(self.vocab_size, d_model)

        self._emb = nn.Sequential(
            self.embedding,
            PositionalEncoding(d_model, dropout_rate)
        )

        self.d_model = d_model
        self.encoder = Encoder(d_model, d_ff, blocks_count, heads_count, dropout_rate)
        self.decoder = Decoder(self.vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def decode_tensor(self, tensor):
        return ' '.join([self.vocab.itos[idx] for idx in tensor])  # if idx.item() not in self.ignore_tokens

    def forward(self, source_inputs, target_inputs, source_mask, target_mask):
        source_embeddings = self._emb(source_inputs)
        target_embeddings = self._emb(target_inputs)

        encoder_output = self.encoder(source_embeddings, source_mask)
        decoder_output = self.decoder(target_embeddings, encoder_output, source_mask, target_mask)
        return decoder_output

    def generate_summary(self, source_inputs, target_inputs, source_mask, target_mask):
        with torch.no_grad():
            outputs = self.generate_only_summary(source_inputs)
            return ([self.decode_tensor(seq) for seq in source_inputs],
                    [self.decode_tensor(seq).split("</s>")[0] + "</s>" for seq in target_inputs],
                    [self.decode_tensor(seq).split("</s>")[0] + "</s>" for seq in
                     outputs])

    def generate_only_summary(self, source_inputs, temperature=1.2):
        batch_size = source_inputs.shape[0]
        max_len = source_inputs.shape[-1]

        generated_tokens = torch.full((batch_size, 1),
                                      self.vocab[self.sos_token],
                                      dtype=torch.long,
                                      device=self.device)

        source_mask = (source_inputs != self.vocab[self.pad_token]).unsqueeze(1)
        source_embeddings = self._emb(source_inputs)
        encoder_output = self.encoder(source_embeddings, source_mask)

        for step in range(max_len):
            tgt_mask = subsequent_mask(generated_tokens.size(1), self.device)
            tgt_mask = tgt_mask & (generated_tokens != self.vocab[self.pad_token]).unsqueeze(1)

            tgt_emb = self._emb(generated_tokens)
            decoder_output = self.decoder(tgt_emb, encoder_output, source_mask, tgt_mask)

            next_token_logits = decoder_output[:, -1, :]

            # Применяем температуру
            next_token_logits = next_token_logits / temperature

            # Запрещаем повторение предыдущего токена
            if generated_tokens.size(1) > 1:
                prev_token = generated_tokens[:, -1]
                next_token_logits[torch.arange(batch_size), prev_token] = -float('inf')

            probs = F.softmax(next_token_logits, dim=-1)

            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Добавляем новый токен к сгенерированной последовательности
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            if (next_token == self.vocab[self.eos_token]).all():
                break

        return generated_tokens

    def old_generate_tokens_for_summary(self, source_inputs):
        batch_size = source_inputs.shape[0]
        max_len = source_inputs.shape[-1]

        # Инициализируем "ложный target" нулями (или <pad>)
        fake_target = torch.full((batch_size, max_len), self.vocab[self.sos_token], dtype=torch.long,
                                 device=self.device)

        # Начинаем с пустого списка токенов
        generated_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)

        # Пропускаем входные данные через энкодер один раз
        source_mask, target_mask = make_mask(source_inputs, fake_target, self.vocab[self.pad_token], self.device)
        source_embeddings = self._emb(source_inputs)
        encoder_output = self.encoder(source_embeddings, source_mask)
        print(encoder_output.shape)
        print(encoder_output[0])  # очень похожи bert
        print(encoder_output[1])  # очень похожи bert

        for step in range(max_len):
            # Обновляем fake_target для текущего шага
            if generated_tokens.shape[1] > 0:
                fake_target[:, :generated_tokens.shape[1]] = generated_tokens

            print(
                f"############################################ {step} TOKEN ################################################")
            print(fake_target)

            # Создаём маску для текущего состояния
            source_mask, target_mask = make_mask(source_inputs, fake_target, self.vocab[self.pad_token], self.device)
            print(target_mask)
            print()

            generated_tokens_emb = self._emb(fake_target)
            decoder_output = self.decoder(generated_tokens_emb, encoder_output, source_mask, target_mask)

            # Берём логиты step токена (какой шаг, такая строчка в decoder_output нам и нужна)
            next_token_logits = decoder_output[:, step, :]
            print("decoder_output")
            print(decoder_output.shape)
            print(decoder_output)
            print("choose", next_token_logits.shape)
            print("choose", next_token_logits)
            # next_token_logits[0][7446] += 1111111
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # Выбираем самый вероятный токен
            print("next: ", next_token)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Останавливаемся, если во всех примерах появился </s>
            if (next_token == self.vocab[self.eos_token]).all():
                break

        return generated_tokens
