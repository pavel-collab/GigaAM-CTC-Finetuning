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
   
    def __init__(self, preprocessor, dataset_part: str="train"):
       self.dataset = load_dataset("google/fleurs", "ru_ru")
       if dataset_part == "train":
        self.dataset = self.dataset['train']
       elif dataset_part == "validation":
        self.dataset = self.dataset['validation']
       else: 
          self.dataset = self.dataset['test']
        
       self.preprocessor = preprocessor
   
    def __len__(self) -> int:
        return len(self.dataset)
   
    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]

        # mel_spec_signal, signal_len = self.preprocessor(
        #    torch.tensor(sample['audio']['array'], dtype=torch.float32),
        #    torch.tensor(sample['num_samples'], dtype=torch.int32)
        # )

        # return {
        #     'audio': mel_spec_signal,
        #     'num_samples': signal_len,
        #     'transcription': sample['transcription'],
        # }

        #! We actually don't need to apply a preprocessor to the data, because
        #! in the this preprocessor is already in vanilla model
        return {
            'audio': torch.tensor(sample['audio']['array'], dtype=torch.float32),
            'num_samples': torch.tensor(sample['num_samples'], dtype=torch.int32),
            'transcription': sample['transcription'],
        }