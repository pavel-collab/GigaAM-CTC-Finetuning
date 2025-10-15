from torch.utils.data import Dataset
import torch
import torchaudio
import json

class AudioDataset(Dataset):
    """
    Датасет для загрузки аудиофайлов и транскрипций
   
    Ожидаемый формат данных:
    - manifest_path: путь к JSON файлу с метаданными
    - Формат JSON: [{"audio_path": "path/to/audio.wav", "text": "транскрипция"}, ...]
    """
   
    def __init__(self, manifest_path: str, preprocessor=None, max_duration: float = 20.0):
        """
        Args:
            manifest_path: путь к файлу манифеста с данными
            preprocessor: предобработчик из GigaAM
            max_duration: максимальная длительность аудио в секундах
        """
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
       
        self.preprocessor = preprocessor
        self.max_duration = max_duration
       
        # Фильтрация слишком длинных аудио
        self.data = [item for item in self.data if self._get_duration(item['audio_path']) <= max_duration]

        logger.info(f"Загружено {len(self.data)} образцов из {manifest_path}")

    def _get_duration(self, audio_path: str) -> float:
        """Получение длительности аудиофайла"""
        try:
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except:
            return 0.0
   
    def __len__(self) -> int:
        return len(self.data)
   
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        audio_path = item['audio_path']
        text = item['text']
       
        # Загрузка аудио с использованием предобработчика GigaAM
        if self.preprocessor:
            waveform, sample_rate = torchaudio.load(audio_path)
            # Преобразование к моно если стерео
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform, sample_rate = torchaudio.load(audio_path)
       
        return {
            'audio': waveform.squeeze(0),
            'text': text,
            'audio_path': audio_path,
            'sample_rate': sample_rate
        }