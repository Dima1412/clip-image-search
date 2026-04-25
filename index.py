"""
Скрипт индексации: проходит по всем картинкам в папке IMAGES_DIR,
прогоняет каждую через CLIP, сохраняет эмбеддинги и пути.
"""

import os
import json
from pathlib import Path

import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm

# ===== Настройки =====
IMAGES_DIR = "images"           # папка с картинками
EMBEDDINGS_FILE = "embeddings.npy"  # куда сохранять векторы
PATHS_FILE = "paths.json"       # куда сохранять пути
MODEL_NAME = "xlm-roberta-base-ViT-B-32"         # архитектура CLIP
PRETRAINED = "laion5b_s13b_b90k"           # на каких данных предобучена

# Расширения файлов, которые считаем картинками
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model():
    """Загружает модель CLIP. При первом запуске скачает ~600 МБ."""
    print("Загружаю модель CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()  # переводим в режим инференса (не обучения)
    return model, preprocess


def get_image_paths(directory):
    """Находит все картинки в папке (включая подпапки)."""
    paths = []
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(path))
    return paths


def encode_image(image_path, model, preprocess):
    """Превращает одну картинку в вектор размерности 512."""
    try:
        image = Image.open(image_path).convert("RGB")
        # preprocess: ресайз, центр-кроп, нормализация — то, что ждёт CLIP
        image_tensor = preprocess(image).unsqueeze(0)  # добавляем batch dim
        
        with torch.no_grad():  # не считаем градиенты — экономим память
            features = model.encode_image(image_tensor)
            # Нормализуем вектор до длины 1, чтобы можно было считать
            # косинусное сходство как обычное скалярное произведение
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.squeeze(0).cpu().numpy()  # вектор numpy
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None


def main():
    # 1. Загружаем модель
    model, preprocess = load_model()
    
    # 2. Находим все картинки
    print(f"\nИщу картинки в папке '{IMAGES_DIR}'...")
    image_paths = get_image_paths(IMAGES_DIR)
    print(f"Найдено картинок: {len(image_paths)}")
    
    if not image_paths:
        print("Картинки не найдены! Положи что-нибудь в папку 'images' и запусти снова.")
        return
    
    # 3. Прогоняем каждую через CLIP
    print("\nИндексирую картинки...")
    embeddings = []
    valid_paths = []
    
    for path in tqdm(image_paths):
        emb = encode_image(path, model, preprocess)
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(path)
    
    # 4. Сохраняем результаты
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(EMBEDDINGS_FILE, embeddings_array)
    
    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_paths, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Готово!")
    print(f"  Векторов сохранено: {len(embeddings)}")
    print(f"  Размерность вектора: {embeddings_array.shape[1]}")
    print(f"  Файлы: {EMBEDDINGS_FILE}, {PATHS_FILE}")


if __name__ == "__main__":
    main()