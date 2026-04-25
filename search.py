"""
Логика поиска: модель CLIP + FAISS-индекс.
Индексирование папки и поиск по тексту/картинке выполняются в памяти.
"""

from pathlib import Path

import torch
import open_clip
import numpy as np
import faiss
from PIL import Image

MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageSearcher:

    def __init__(self):
        print("Загружаю модель CLIP...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model.eval()
        self.paths = []
        self.index = None

    @property
    def has_index(self):
        return self.index is not None

    def build_index(self, folder_path, on_progress=None):
        """
        Сканирует папку, кодирует изображения CLIP и строит FAISS-индекс.
        on_progress(current, total) вызывается после каждого обработанного файла.
        Возвращает количество успешно проиндексированных изображений.
        """
        image_paths = [
            p for p in Path(folder_path).rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(f"В папке '{folder_path}' не найдено изображений")

        embeddings, valid_paths = [], []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            if on_progress:
                on_progress(i, total)
            try:
                img = Image.open(path).convert("RGB")
                tensor = self.preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    feat = self.model.encode_image(tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.squeeze(0).cpu().numpy())
                valid_paths.append(str(path))
            except Exception:
                pass

        if on_progress:
            on_progress(total, total)

        arr = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(arr.shape[1])
        self.index.add(arr)
        self.paths = valid_paths
        return len(valid_paths)

    def _encode_text(self, text):
        tokens = self.tokenizer([text])
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    def _encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    def _search(self, query_vector, top_k):
        scores, indices = self.index.search(query_vector, top_k)
        return [
            {"path": self.paths[idx], "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
        ]

    def search_by_text(self, text, top_k=5):
        templates = [
            text,
            f"фотография: {text}",
            f"a photo of {text}",
            f"изображение, на котором {text}",
        ]
        vectors = [self._encode_text(t) for t in templates]
        avg = np.mean(vectors, axis=0)
        avg = (avg / np.linalg.norm(avg)).astype(np.float32)
        return self._search(avg, top_k)

    def search_by_image(self, image, top_k=5):
        return self._search(self._encode_image(image), top_k)
