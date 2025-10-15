from typing import Dict, List
import torch

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Функция для объединения батча с паддингом аудио до одинаковой длины
    """
    # Находим максимальную длину в батче
    max_length = max([item['audio'].shape[0] for item in batch])
   
    # Паддинг аудио
    audios = []
    audio_lengths = []
    texts = []
   
    for item in batch:
        audio = item['audio']
        audio_length = audio.shape[0]
       
        # Паддинг нулями
        if audio_length < max_length:
            padding = torch.zeros(max_length - audio_length)
            audio = torch.cat([audio, padding])
       
        audios.append(audio)
        audio_lengths.append(audio_length)
        texts.append(item['text'])
   
    return {
        'audios': torch.stack(audios),
        'audio_lengths': torch.tensor(audio_lengths),
        'texts': texts
    }