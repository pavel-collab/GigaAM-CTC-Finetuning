import re
import string

def convert_arabic_number_to_words(number_str):
    """
    Преобразует арабское число в слова
    """
    try:
        num = int(number_str)
    except ValueError:
        return number_str
    
    # Базовые словари
    units = ['', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
    teens = ['десять', 'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 
             'пятнадцать', 'шестнадцать', 'семнадцать', 'восемнадцать', 'девятнадцать']
    tens = ['', '', 'двадцать', 'тридцать', 'сорок', 'пятьдесят', 
            'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто']
    hundreds = ['', 'сто', 'двести', 'триста', 'четыреста', 'пятьсот', 
                'шестьсот', 'семьсот', 'восемьсот', 'девятьсот']
    
    if num == 0:
        return 'ноль'
    
    words = []
    
    # Обрабатываем тысячи
    if num >= 1000:
        thousands = num // 1000
        if thousands == 1:
            words.append('тысяча')
        elif thousands in [2, 3, 4]:
            words.append(units[thousands] + ' тысячи')
        else:
            words.append(convert_arabic_number_to_words(str(thousands)) + ' тысяч')
        num %= 1000
    
    # Обрабатываем сотни
    if num >= 100:
        words.append(hundreds[num // 100])
        num %= 100
    
    # Обрабатываем десятки и единицы
    if num >= 20:
        words.append(tens[num // 10])
        if num % 10 > 0:
            words.append(units[num % 10])
    elif num >= 10:
        words.append(teens[num - 10])
    elif num > 0:
        words.append(units[num])
    
    return ' '.join(words)

def convert_roman_number_to_words(roman_str):
    """
    Преобразует римское число в слова
    """
    roman_numerals = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 
        'C': 100, 'D': 500, 'M': 1000
    }
    
    roman_str = roman_str.upper()
    total = 0
    prev_value = 0
    
    for char in reversed(roman_str):
        if char not in roman_numerals:
            return roman_str
        
        value = roman_numerals[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return convert_arabic_number_to_words(str(total))

def replace_latin_with_russian(text):
    """
    Заменяет латинские буквы на похожие русские
    """
    latin_to_russian = {
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 
        'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й', 
        'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о', 
        'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 
        'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'у', 'z': 'з'
    }
    
    for latin, russian in latin_to_russian.items():
        text = text.replace(latin, russian)
        text = text.replace(latin.upper(), russian.upper())
    
    return text

def preprocess_text(text):
    """
    Предобрабатывает текст по заданным правилам:
    1. Оставляет только русские буквы в нижнем регистре
    2. Заменяет знаки препинания на пробелы
    3. Заменяет ё на е
    4. Преобразует числа в слова
    5. Заменяет английские буквы на русские
    """
    
    # Шаг 1: Заменяем ё на е
    text = text.replace('ё', 'е').replace('Ё', 'е')
    
    # Шаг 2: Заменяем латинские буквы на русские
    text = replace_latin_with_russian(text)
    
    # Шаг 3: Преобразуем римские цифры
    # Ищем римские цифры (от I до MMMCMXCIX)
    roman_pattern = r'\b[IVXLCDM]+\b'
    text = re.sub(roman_pattern, lambda m: convert_roman_number_to_words(m.group()), text, flags=re.IGNORECASE)
    
    # Шаг 4: Преобразуем арабские цифры
    # Ищем целые числа
    arabic_pattern = r'\b\d+\b'
    text = re.sub(arabic_pattern, lambda m: convert_arabic_number_to_words(m.group()), text)
    
    # Шаг 5: Заменяем все знаки препинания на пробелы
    punctuation_chars = string.punctuation + '—–«»„“‚‘'
    for char in punctuation_chars:
        text = text.replace(char, ' ')
    
    # Шаг 6: Приводим к нижнему регистру
    text = text.lower()
    
    # Шаг 7: Удаляем все символы, кроме русских букв и пробелов
    text = re.sub(r'[^а-я\s]', '', text)
    
    # Шаг 8: Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text