from torch.utils.data import Dataset
import torch
from datasets import load_dataset
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#! Сейчас скачивание целевого датасета происходит при инициализации класса
#! Такой подход чреват большим накладными расходами на загрузку данных, многократным скачиванием и
#! Хранением большого количетсва данных в оперативной памяти.
#! Возможно, стоит рассмотреть подход с отдельной загрузкой целевого датасета на локалку и
#! Извлечение его через манифесты
class AudioDataset(Dataset):
    """
    Датасет для загрузки аудиофайлов и транскрипций
   
    Ожидаемый формат данных:
    - manifest_path: путь к JSON файлу с метаданными
    - Формат JSON: [{"audio_path": "path/to/audio.wav", "text": "транскрипция"}, ...]
    """
   
    def __init__(self, dataset_part, normalize_fn: str="train"):
        self.dataset = load_dataset("google/fleurs", "ru_ru")
        
        if dataset_part == "train":
            self.dataset = self.dataset['train']
        elif dataset_part == "validation":
            self.dataset = self.dataset['validation']
        else: 
            self.dataset = self.dataset['test']
        
        self.normalize_fn = normalize_fn

        self.indices = []
        for idx in range(len(self.dataset)):
            self.indices.append(idx)

   
    def __len__(self) -> int:
        return len(self.indices)
   
    def __getitem__(self, idx: int) -> Dict:
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]
        
        # Аудио
        audio_array = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']
        wav = torch.from_numpy(audio_array).float()
        
        # Нормализация к моно
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        
        # Текст
        text = self.normalize_fn(sample['transcription'])
        
        return wav, text, sampling_rate