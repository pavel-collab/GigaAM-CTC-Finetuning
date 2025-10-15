from typing import List
import logging
from tqdm import tqdm
from pathlib import Path
import json
import torchaudio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_common_voice(
    cv_dir: str,
    output_dir: str,
    splits: List[str] = ['train', 'dev', 'test']
):
    """
    Конвертация датасета Common Voice в формат манифеста
   
    Args:
        cv_dir: директория с Common Voice
        output_dir: директория для сохранения манифестов
        splits: список split'ов для обработки
    """
    import csv
   
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
   
    for split in splits:
        tsv_file = Path(cv_dir) / f"{split}.tsv"
        if not tsv_file.exists():
            logger.warning(f"Файл не найден: {tsv_file}")
            continue
       
        manifest = []
       
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
           
            for row in tqdm(reader, desc=f"Обработка {split}"):
                audio_filename = row['path']
                text = row['sentence']
               
                audio_path = Path(cv_dir) / 'clips' / audio_filename
               
                if not audio_path.exists():
                    continue
               
                try:
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                   
                    manifest.append({
                        'audio_path': str(audio_path.absolute()),
                        'text': text,
                        'duration': duration,
                        'sample_rate': info.sample_rate,
                    })
                except Exception as e:
                    logger.warning(f"Ошибка при обработке {audio_path}: {e}")
       
        # Сохранение манифеста
        manifest_file = output_path / f"{split}_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
       
        logger.info(f"Сохранен {split} манифест: {manifest_file} ({len(manifest)} записей)")

def convert_librispeech(
    librispeech_dir: str,
    output_manifest: str,
):
    """
    Конвертация датасета LibriSpeech в формат манифеста
   
    Args:
        librispeech_dir: директория с LibriSpeech
        output_manifest: путь для сохранения манифеста
    """
    manifest = []
   
    librispeech_path = Path(librispeech_dir)
   
    # Поиск всех .trans.txt файлов
    trans_files = list(librispeech_path.rglob("*.trans.txt"))
   
    logger.info(f"Найдено {len(trans_files)} файлов с транскрипциями")
   
    for trans_file in tqdm(trans_files, desc="Обработка LibriSpeech"):
        audio_dir = trans_file.parent
       
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
               
                audio_id, text = parts
                audio_path = audio_dir / f"{audio_id}.flac"
               
                if not audio_path.exists():
                    continue
               
                try:
                    info = torchaudio.info(str(audio_path))
                    duration = info.num_frames / info.sample_rate
                   
                    manifest.append({
                        'audio_path': str(audio_path.absolute()),
                        'text': text.lower(),
                        'duration': duration,
                        'sample_rate': info.sample_rate,
                    })
                except Exception as e:
                    logger.warning(f"Ошибка при обработке {audio_path}: {e}")
   
    # Сохранение манифеста
    output_path = Path(output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
   
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
   
    logger.info(f"Манифест сохранен: {output_path} ({len(manifest)} записей)")

def analyze_manifest(manifest_path: str):
    """
    Анализ манифеста датасета
   
    Args:
        manifest_path: путь к манифесту
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
   
    if not manifest:
        logger.error("Манифест пуст!")
        return
   
    durations = [item['duration'] for item in manifest]
    text_lengths = [len(item['text']) for item in manifest]
   
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
   
    avg_text_len = sum(text_lengths) / len(text_lengths)
   
    print(f"\n{'='*50}")
    print(f"Анализ манифеста: {manifest_path}")
    print(f"{'='*50}")
    print(f"Всего образцов: {len(manifest)}")
    print(f"Общая длительность: {total_duration / 3600:.2f} часов")
    print(f"Средняя длительность: {avg_duration:.2f} секунд")
    print(f"Минимальная длительность: {min_duration:.2f} секунд")
    print(f"Максимальная длительность: {max_duration:.2f} секунд")
    print(f"Средняя длина текста: {avg_text_len:.1f} символов")
    print(f"{'='*50}\n")
   
    # Распределение по длительности
    bins = [0, 2, 5, 10, 15, 20, float('inf')]
    bin_labels = ['0-2s', '2-5s', '5-10s', '10-15s', '15-20s', '20s+']

    print("Распределение по длительности:")
    for i in range(len(bins) - 1):
        count = sum(1 for d in durations if bins[i] <= d < bins[i+1])
        percentage = (count / len(durations)) * 100
        print(f"  {bin_labels[i]}: {count} ({percentage:.1f}%)")