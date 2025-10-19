from gigaam import GigaAMASR
import gigaam
import os
import torch
import re
from typing import List, Dict

def get_model_vocab(model):
    model_vocab = {sym: idx for idx, sym in enumerate(model.decoding.tokenizer.vocab)}
    return model_vocab

def preprocess_text(text):
    """
    Предобрабатывает текст по заданным правилам:
    1. Оставляет только символы русского алфавита и пробелы
    2. Заменяет дефисы на пробелы
    3. Заменяет латинские буквы на близкие русские
    4. Заменяет арабские и римские цифры на слова
    """
    
    # Словарь замены латинских букв на русские
    latin_to_russian = {
        'a': 'а', 'A': 'А',
        'b': 'б', 'B': 'Б',
        'c': 'к', 'C': 'К',
        'd': 'д', 'D': 'Д',
        'e': 'е', 'E': 'Е',
        'f': 'ф', 'F': 'Ф',
        'g': 'г', 'G': 'Г',
        'h': 'х', 'H': 'Х',
        'i': 'и', 'I': 'И',
        'j': 'й', 'J': 'Й',
        'k': 'к', 'K': 'К',
        'l': 'л', 'L': 'Л',
        'm': 'м', 'M': 'М',
        'n': 'н', 'N': 'Н',
        'o': 'о', 'O': 'О',
        'p': 'п', 'P': 'П',
        'q': 'к', 'Q': 'К',
        'r': 'р', 'R': 'Р',
        's': 'с', 'S': 'С',
        't': 'т', 'T': 'Т',
        'u': 'у', 'U': 'У',
        'v': 'в', 'V': 'В',
        'w': 'в', 'W': 'В',
        'x': 'кс', 'X': 'Кс',
        'y': 'у', 'Y': 'У',
        'z': 'з', 'Z': 'З'
    }
    
    # Словарь для замены арабских цифр
    digit_to_word = {
        '0': 'ноль',
        '1': 'один',
        '2': 'два',
        '3': 'три',
        '4': 'четыре',
        '5': 'пять',
        '6': 'шесть',
        '7': 'семь',
        '8': 'восемь',
        '9': 'девять'
    }
    
    # Словарь для замены римских цифр
    roman_to_word = {
        'I': 'один', 'II': 'два', 'III': 'три', 'IV': 'четыре', 'V': 'пять',
        'VI': 'шесть', 'VII': 'семь', 'VIII': 'восемь', 'IX': 'девять', 'X': 'десять',
        'XI': 'одиннадцать', 'XII': 'двенадцать', 'XIII': 'тринадцать',
        'XIV': 'четырнадцать', 'XV': 'пятнадцать', 'XVI': 'шестнадцать',
        'XVII': 'семнадцать', 'XVIII': 'восемнадцать', 'XIX': 'девятнадцать',
        'XX': 'двадцать'
    }
    
    # Приводим текст к нижнему регистру для удобства обработки
    text = text.lower()
    
    # Заменяем римские цифры (обрабатываем сначала перед другими преобразованиями)
    for roman, word in sorted(roman_to_word.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r'\b' + roman + r'\b', word, text, flags=re.IGNORECASE)
    
    # Заменяем латинские буквы на русские
    for latin, russian in latin_to_russian.items():
        text = text.replace(latin, russian)
    
    # Заменяем арабские цифры
    for digit, word in digit_to_word.items():
        text = text.replace(digit, word)
    
    # Заменяем дефисы на пробелы
    text = text.replace('-', ' ')
    
    # Удаляем все символы, кроме русских букв и пробелов
    text = re.sub(r'[^а-я\s]', '', text, flags=re.IGNORECASE)
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def import_gigaam_model(model_type: str="ctc", device: str="cpu"):
    CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
    model_name, model_path = gigaam._download_model(model_type, CACHE_DIR)

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    ckpt["cfg"].encoder.flash_attn = False
    model = GigaAMASR(ckpt['cfg'])

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.eval()

    if device.type != "cpu":
        model.encoder = model.encoder.half()

    return model

def get_gigaam_logprobs(model, wav_batch, wav_lengths, return_transcriptions=False):
    wav_batch = wav_batch.to(model._device)
    wav_lengths = wav_lengths.to(model._device)

    encoded, encoded_len = model.forward(wav_batch, wav_lengths)

    logprobs = model.head(encoded)

    if return_transcriptions:
        transcriptions = model.decoding.decode(model.head, encoded, encoded_len)
        return logprobs, encoded_len, transcriptions
    else:
        return logprobs, encoded_len

def get_texts_idxs(texts: List[str], model_vocab: Dict[str][str]) -> torch.Tensor:
  texts_idxs = []
  for text in texts:
    print(f"[DEBUG] {text}")

    text = preprocess_text(text)

    print(f"[DEBUG] preprocessed text: {text}")

    text_idxs = [model_vocab[sym] for sym in text]
    texts_idxs.append(text_idxs)

  return torch.tensor(texts_idxs, dtype=torch.int)