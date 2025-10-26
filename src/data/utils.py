from src.models.utils import text_to_indices

import torch

class FunctionFactory:
    def __init__(self, model_vocabular=None):
        self.model_vocabular = model_vocabular
    
    def create_default_collate_function(self):
        # Замыкание - функция "запоминает" состояние объекта
        def collate_fn(batch):
            # Для полей с разной длиной (например, audio) нужно добавить паддинг
            audio = [item['audio'] for item in batch]
            audio_padded = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

            return audio_padded, torch.stack([item['num_samples'] for item in batch]), [item['transcription'] for item in batch]
        return collate_fn
    
    def create_advanced_collate_function(self):
        if self.model_vocabular is None:
            raise Exception
        def collate_fn(batch):
            wavs, texts, sampling_rates = zip(*batch)

            wav_lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
            max_len = int(wav_lengths.max())
            B = len(wavs)

            wav_batch = torch.zeros(B, max_len, dtype=torch.float32)
            for i, w in enumerate(wavs):
                wav_batch[i, :w.shape[0]] = w.float()

            target_lists = [text_to_indices(t, self.model_vocabular) for t in texts]
            target_lengths = torch.tensor([len(t) for t in target_lists], dtype=torch.long)
            targets = torch.tensor([idx for seq in target_lists for idx in seq], dtype=torch.long)

            return wav_batch, wav_lengths, targets, target_lengths, texts
        return collate_fn
