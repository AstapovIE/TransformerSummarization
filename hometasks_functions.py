from rouge import Rouge
from model.Masker import convert_batch

from matplotlib import pyplot as plt
import seaborn as sns


def task1(model, test_iter, num_examples, device, output_file):
    """   Демонстрация результата для 5 примеров из теста и 5 собственных примеров.   """
    print("Выполняем генерацию суммаризации для примеров...")
    rouge = Rouge()

    def evaluate_rouge(preds, refs):
        scores = rouge.get_scores(preds, refs, avg=True)
        return scores

    test_batch = next(iter(test_iter))
    source_inputs, target_inputs, source_mask, target_mask = convert_batch(test_batch, device)

    source_text = [model.decode_tensor(seq) for seq in source_inputs[:num_examples]]
    target_text = [model.decode_tensor(seq) for seq in target_inputs[:num_examples]]

    gen_tokens = model.generate_only_summary(source_inputs[:num_examples])#, source_mask[:num_examples])
    output_text = [model.decode_tensor(seq).split("</s>")[0] + "</s>" for seq in gen_tokens]

    with open(f"{output_file}.txt", 'w', encoding='utf-8') as f:
        for i in range(num_examples):
            f.write(f"Исходный текст: {source_text[i]}\n")
            f.write(f"Ожидаемая суммаризация: {target_text[i]}\n")
            f.write(f"Суммаризация от модели: {output_text[i]}\n")
            f.write(f"ROUGE Scores: {evaluate_rouge(output_text[i], target_text[i])}\n\n")

    print(f"Результаты записаны в {output_file}.txt")


def plot_attention_grid(attn_weights, input_tokens, output_file, num_first_tokens=10):
    """
    Визуализация attention для всех 8 голов в одной фигуре.
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))  # 2 строки, 4 столбца
    axes = axes.flatten()  # Разворачиваем в список для удобного доступа

    for i in range(8):  # 8 attention голов
        ax = axes[i]
        sns.heatmap(attn_weights[i, :num_first_tokens, :num_first_tokens].cpu().detach().numpy(),
                    xticklabels=input_tokens[:num_first_tokens],
                    yticklabels=input_tokens[:num_first_tokens],
                    cmap="Blues",
                    cbar=False,
                    ax=ax)
        ax.set_title(f"Head {i + 1}")
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def task3(model, test_iter, num_examples, device, output_file):
    """  Визуализация механизма attention после обучения модели на конкретном примере  """
    print("Строим тепловую карту для визуализации весов механизма внимания в энкодере...")

    for i in range(num_examples):
        batch = next(iter(test_iter))
        source_inputs, _, source_mask, _ = convert_batch(batch, device)

        # Прогон через модель, чтобы attention-матрицы сохранились в _attn_probs
        source_inputs_float = model._emb(source_inputs)

        encoder_outputs = model.encoder(source_inputs_float, source_mask)

        input_tokens = [model.vocab.itos[idx] for idx in source_inputs[0]
                        if idx not in {model.vocab.stoi['<s>'],
                                       model.vocab.stoi['</s>'],
                                       model.vocab.stoi['<pad>']}]

        # Извлекаем attention-матрицы (они должны быть сохранены в _attn_probs)
        attn_weights = model.encoder._blocks[0]._self_attn._attn_probs
        attn_weights = attn_weights[0]  # Берем первую последовательность в батче

        plot_attention_grid(attn_weights, input_tokens, f"{output_file}_{i}.png")
