import torch

# def collate_fn(batch):
#     # Для полей с разной длиной (например, audio) нужно добавить паддинг
#     audio = [item['audio'] for item in batch]
#     audio_padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

#     return audio_padded, torch.stack([item['num_samples'] for item in batch]), [item['transcription'] for item in batch]

def text_to_indices(text, char_to_idx):
    """Конвертация текста в индексы согласно официальному словарю"""
    indices = []
    for ch in text:
        if ch in char_to_idx:
            indices.append(char_to_idx[ch])
        else:
            print(f"Предупреждение: символ '{ch}' не найден в словаре")
    return indices

# wrapper, потому что оригинальная сигнатура collate_fn принимает 1 аргумент, но мы эту проблемку решим на месте с помощью lambda функции
def collate_fn_wrapper(batch, char2idx):
    wavs, texts, sampling_rates = zip(*batch)

    wav_lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_len = int(wav_lengths.max())
    B = len(wavs)

    wav_batch = torch.zeros(B, max_len, dtype=torch.float32)
    for i, w in enumerate(wavs):
        wav_batch[i, :w.shape[0]] = w.float()

    target_lists = [text_to_indices(t, char2idx) for t in texts]
    target_lengths = torch.tensor([len(t) for t in target_lists], dtype=torch.long)
    targets = torch.tensor([idx for seq in target_lists for idx in seq], dtype=torch.long)

    return wav_batch, wav_lengths, targets, target_lengths, texts