import torch

def collate_fn(batch):
    # Для полей с разной длиной (например, audio) нужно добавить паддинг
    audio = [item['audio'] for item in batch]
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

    return audio_padded, torch.stack([item['num_samples'] for item in batch]), [item['transcription'] for item in batch]