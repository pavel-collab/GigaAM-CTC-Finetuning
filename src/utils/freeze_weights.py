def freeze_model_completely(model):
    # Заморозить весь препроцессор - это общие аудио-фичи
    for param in model.preprocessor.parameters():
        param.requires_grad = False
    
    # Заморозить весь энкодер
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Оставить размороженной только голову
    for param in model.head.parameters():
        param.requires_grad = True

def freeze_model_selective(model, num_frozen_layers=12):
    """Заморозить первые N слоев энкодера"""
    
    # 1. Всегда замораживаем препроцессор
    for param in model.preprocessor.parameters():
        param.requires_grad = False
    
    # 2. Заморозить subsampling в энкодере
    for param in model.encoder.pre_encode.parameters():
        param.requires_grad = False
    
    # 3. Заморозить позиционные эмбеддинги
    for param in model.encoder.pos_enc.parameters():
        param.requires_grad = False
    
    # 4. Заморозить первые N слоев Conformer
    for i, layer in enumerate(model.encoder.layers):
        if i < num_frozen_layers:  # Замораживаем первые 12 из 16 слоев
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    
    # 5. Голова всегда разморожена
    for param in model.head.parameters():
        param.requires_grad = True

def freeze_by_components(model):
    """Заморозка по типам компонентов"""
    
    # Препроцессор - всегда заморожен
    for param in model.preprocessor.parameters():
        param.requires_grad = False
    
    # Энкодер - частичная заморозка
    for name, param in model.encoder.named_parameters():
        if 'pre_encode' in name or 'pos_enc' in name:
            param.requires_grad = False  # Субсемплинг и позиционные эмбеддинги
        elif any(f'layers.{i}.' in name for i in range(8)):  # Первые 8 слоев
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Голова - всегда разморожена
    for param in model.head.parameters():
        param.requires_grad = True