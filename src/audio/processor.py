import json
import torchaudio
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparator:
    """
    Класс для подготовки датасета к обучению
    """
   
    def __init__(
        self,
        audio_dir: str,
        transcripts_file: str,
        output_manifest: str,
        min_duration: float = 0.5,
        max_duration: float = 20.0,
        sample_rate: int = 16000,
    ):
        """
        Args:
            audio_dir: директория с аудиофайлами
            transcripts_file: файл с транскрипциями (формат: путь|текст или JSON)
            output_manifest: путь для сохранения манифеста
            min_duration: минимальная длительность аудио (секунды)
            max_duration: максимальная длительность аудио (секунды)
            sample_rate: целевая частота дискретизации
        """
        self.audio_dir = Path(audio_dir)
        self.transcripts_file = transcripts_file
        self.output_manifest = output_manifest
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sample_rate = sample_rate

    def load_transcripts(self) -> Dict[str, str]:
        """
        Загрузка транскрипций из файла
       
        Поддерживаемые форматы:
        1. TSV/CSV: audio_path\ttext или audio_path|text
        2. JSON: [{"audio_path": "...", "text": "..."}, ...]
        """
        transcripts = {}
       
        ext = Path(self.transcripts_file).suffix.lower()
       
        if ext == '.json':
            # JSON формат
            with open(self.transcripts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    transcripts[item['audio_path']] = item['text']
        else:
            # TSV/TXT формат
            with open(self.transcripts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                   
                    # Пробуем разные разделители
                    if '|' in line:
                        parts = line.split('|', 1)
                    elif '\t' in line:
                        parts = line.split('\t', 1)
                    else:
                        parts = line.split(' ', 1)
                   
                    if len(parts) == 2:
                        audio_path, text = parts
                        transcripts[audio_path.strip()] = text.strip()
        
        logger.info(f"Загружено {len(transcripts)} транскрипций")
        return transcripts
    
    def get_audio_info(self, audio_path: str) -> Optional[Dict]:
        """
        Получение информации об аудиофайле
       
        Args:
            audio_path: путь к аудиофайлу
           
        Returns:
            словарь с информацией или None при ошибке
        """
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
           
            return {
                'duration': duration,
                'sample_rate': info.sample_rate,
                'num_channels': info.num_channels,
                'num_frames': info.num_frames,
            }
        except Exception as e:
            logger.warning(f"Ошибка при чтении {audio_path}: {e}")
            return None
        
    def validate_audio(self, audio_path: str, info: Dict) -> bool:
        """
        Проверка аудио на соответствие критериям
       
        Args:
            audio_path: путь к файлу
            info: информация об аудио
           
        Returns:
            True если аудио проходит проверку
        """
        duration = info['duration']
       
        # Проверка длительности
        if duration < self.min_duration:
            logger.debug(f"Пропуск {audio_path}: слишком короткое ({duration:.2f}s)")
            return False
       
        if duration > self.max_duration:
            logger.debug(f"Пропуск {audio_path}: слишком длинное ({duration:.2f}s)")
            return False
       
        return True
    
    def prepare_manifest(self) -> List[Dict]:
        """
        Создание манифеста датасета
       
        Returns:
            список записей для манифеста
        """
        transcripts = self.load_transcripts()
        manifest = []
       
        logger.info("Обработка аудиофайлов...")
       
        # Обработка каждой транскрипции
        for audio_filename, text in tqdm(transcripts.items(), desc="Обработка"):
            # Формирование полного пути
            audio_path = self.audio_dir / audio_filename
           
            if not audio_path.exists():
                # Пробуем найти в подпапках
                found = list(self.audio_dir.rglob(audio_filename))
                if found:
                    audio_path = found[0]
                else:
                    logger.warning(f"Файл не найден: {audio_path}")
                    continue
           
            # Получение информации об аудио
            info = self.get_audio_info(str(audio_path))
            if info is None:
                continue
           
            # Валидация
            if not self.validate_audio(str(audio_path), info):
                continue
           
            # Добавление в манифест
            manifest.append({
                'audio_path': str(audio_path.absolute()),
                'text': text,
                'duration': info['duration'],
                'sample_rate': info['sample_rate'],
            })
        
        logger.info(f"Создано записей в манифесте: {len(manifest)}")
        return manifest
    
    def save_manifest(self, manifest: List[Dict]):
        """
        Сохранение манифеста в JSON файл
       
        Args:
            manifest: список записей манифеста
        """
        output_path = Path(self.output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
       
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
       
        logger.info(f"Манифест сохранен: {output_path}")
       
        # Вывод статистики
        total_duration = sum(item['duration'] for item in manifest)
        avg_duration = total_duration / len(manifest) if manifest else 0
       
        logger.info(f"Статистика датасета:")
        logger.info(f"  Всего образцов: {len(manifest)}")
        logger.info(f"  Общая длительность: {total_duration / 3600:.2f} часов")
        logger.info(f"  Средняя длительность: {avg_duration:.2f} секунд")

    def split_manifest(
        self,
        manifest: List[Dict],
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
    ) -> tuple:
        """
        Разделение манифеста на train/val
       
        Args:
            manifest: полный манифест
            train_ratio: доля тренировочных данных
            val_ratio: доля валидационных данных
           
        Returns:
            (train_manifest, val_manifest)
        """
        import random
        random.seed(42)
       
        # Перемешивание
        manifest_shuffled = manifest.copy()
        random.shuffle(manifest_shuffled)
       
        # Разделение
        total = len(manifest_shuffled)
        train_size = int(total * train_ratio)
       
        train_manifest = manifest_shuffled[:train_size]
        val_manifest = manifest_shuffled[train_size:]
       
        logger.info(f"Разделение: train={len(train_manifest)}, val={len(val_manifest)}")
       
        return train_manifest, val_manifest
    
    def run(self, split: bool = True):
        """
        Запуск полного процесса подготовки данных
       
        Args:
            split: разделить на train/val
        """
        manifest = self.prepare_manifest()
       
        if split:
            train_manifest, val_manifest = self.split_manifest(manifest)
           
            # Сохранение train
            train_path = str(Path(self.output_manifest).parent / "train_manifest.json")
            self.output_manifest = train_path
            self.save_manifest(train_manifest)
           
            # Сохранение val
            val_path = str(Path(self.output_manifest).parent / "val_manifest.json")
            self.output_manifest = val_path
            self.save_manifest(val_manifest)
        else:
            self.save_manifest(manifest)
    