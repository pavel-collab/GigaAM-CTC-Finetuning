from src.audio.utils import convert_common_voice, convert_librispeech, analyze_manifest
from src.audio.processor import DatasetPreparator

import argparse

def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для обучения GigaAM")
   
    subparsers = parser.add_subparsers(dest='command', help='Команды')
   
    # Команда prepare
    prepare_parser = subparsers.add_parser('prepare', help='Подготовка кастомного датасета')
    prepare_parser.add_argument('--audio_dir', type=str, required=True,
                               help='Директория с аудиофайлами')
    prepare_parser.add_argument('--transcripts', type=str, required=True,
                               help='Файл с транскрипциями')
    prepare_parser.add_argument('--output', type=str, required=True,
                               help='Путь для сохранения манифеста')
    prepare_parser.add_argument('--min_duration', type=float, default=0.5,
                               help='Минимальная длительность (сек)')
    prepare_parser.add_argument('--max_duration', type=float, default=20.0,
                               help='Максимальная длительность (сек)')
    prepare_parser.add_argument('--no_split', action='store_true',
                               help='Не разделять на train/val')
   
    # Команда convert_cv
    cv_parser = subparsers.add_parser('convert_cv', help='Конвертация Common Voice')
    cv_parser.add_argument('--cv_dir', type=str, required=True,
                          help='Директория Common Voice')
    cv_parser.add_argument('--output_dir', type=str, required=True,
                          help='Директория для манифестов')
   
    # Команда convert_librispeech
    ls_parser = subparsers.add_parser('convert_librispeech', help='Конвертация LibriSpeech')
    ls_parser.add_argument('--librispeech_dir', type=str, required=True,
                          help='Директория LibriSpeech')
    ls_parser.add_argument('--output', type=str, required=True,
                          help='Путь для сохранения манифеста')
   
    # Команда analyze
    analyze_parser = subparsers.add_parser('analyze', help='Анализ манифеста')
    analyze_parser.add_argument('--manifest', type=str, required=True,
                               help='Путь к манифесту')
   
    args = parser.parse_args()

    if args.command == 'prepare':
        preparator = DatasetPreparator(
            audio_dir=args.audio_dir,
            transcripts_file=args.transcripts,
            output_manifest=args.output,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        preparator.run(split=not args.no_split)
   
    elif args.command == 'convert_cv':
        convert_common_voice(
            cv_dir=args.cv_dir,
            output_dir=args.output_dir,
        )
   
    elif args.command == 'convert_librispeech':
        convert_librispeech(
            librispeech_dir=args.librispeech_dir,
            output_manifest=args.output,
        )
   
    elif args.command == 'analyze':
        analyze_manifest(args.manifest)
   
    else:
        parser.print_help()
        
if __name__ == "__main__":
    main()