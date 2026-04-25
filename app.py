"""
Streamlit-интерфейс для поиска похожих изображений через CLIP.
Запуск: streamlit run app.py
"""

import tkinter as tk
from tkinter import filedialog

import streamlit as st
from PIL import Image

from search import ImageSearcher

st.set_page_config(page_title="CLIP Image Search", page_icon="🔍", layout="wide")


@st.cache_resource
def load_model():
    return ImageSearcher()


def pick_folder():
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        folder = filedialog.askdirectory(parent=root, title="Выбери папку с изображениями")
        root.destroy()
        return folder or None
    except Exception:
        return None


def run_indexing(searcher, folder_path):
    bar = st.progress(0.0, text="Начинаю индексацию...")

    def on_progress(current, total):
        bar.progress(
            current / total if total else 1.0,
            text=f"Обрабатываю {current} / {total}...",
        )

    try:
        count = searcher.build_index(folder_path, on_progress=on_progress)
        bar.progress(1.0, text=f"Готово! Проиндексировано: {count}")
        return count
    except ValueError as e:
        bar.empty()
        st.error(str(e))
        return 0


def display_results(results, min_score=0.0):
    filtered = [r for r in results if r["score"] >= min_score]
    if not filtered:
        st.warning("Ничего не нашлось выше заданного порога. Попробуй снизить порог в настройках.")
        return

    st.subheader(f"Результаты: {len(filtered)}")
    cols_per_row = 3
    for i in range(0, len(filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, result in zip(cols, filtered[i : i + cols_per_row]):
            with col:
                try:
                    st.image(
                        Image.open(result["path"]),
                        caption=f"Сходство: {result['score']:.3f}",
                        use_container_width=True,
                    )
                    st.caption(f"📁 {result['path']}")
                except Exception as e:
                    st.error(f"Не удалось открыть файл: {e}")


def main():
    st.title("🔍 Поиск похожих изображений")
    st.caption("Поиск по коллекции изображений с помощью CLIP.")

    with st.spinner("Загружаю модель..."):
        searcher = load_model()

    top_k, min_score = 6, 0.0

    # ===== Боковая панель =====
    with st.sidebar:
        st.header("📁 Коллекция")

        if "folder_path" not in st.session_state:
            st.session_state.folder_path = ""
        if "indexed_folder" not in st.session_state:
            st.session_state.indexed_folder = ""

        if st.button("📂 Выбрать папку"):
            picked = pick_folder()
            if picked:
                if picked != st.session_state.folder_path:
                    st.session_state.indexed_folder = ""
                st.session_state.folder_path = picked
                st.rerun()

        old_path = st.session_state.folder_path
        folder_path = st.text_input("Или введи путь вручную:", value=old_path)
        if folder_path != old_path:
            st.session_state.indexed_folder = ""
        st.session_state.folder_path = folder_path

        already_indexed = (
            bool(folder_path)
            and st.session_state.indexed_folder == folder_path
            and searcher.has_index
        )

        if already_indexed:
            st.success(f"✅ {len(searcher.paths)} изображений в индексе")

        if folder_path:
            if st.button("⚡ Индексировать", type="primary", disabled=already_indexed):
                count = run_indexing(searcher, folder_path)
                if count > 0:
                    st.session_state.indexed_folder = folder_path
                    st.rerun()
        else:
            st.info("Выбери папку с изображениями")

        if searcher.has_index:
            st.divider()
            st.header("⚙️ Настройки")
            top_k = st.slider("Количество результатов", 1, 20, 6)
            min_score = st.slider("Минимальное сходство", 0.0, 0.5, 0.0, 0.01)
            st.caption("Оценки CLIP обычно в диапазоне 0.15–0.35.")

    # ===== Основной контент =====
    if not searcher.has_index:
        st.info("👈 Выбери папку с изображениями и нажми «Индексировать»")
        return

    tab_text, tab_image = st.tabs(["🔤 Поиск по тексту", "🖼️ Поиск по картинке"])

    with tab_text:
        query = st.text_input(
            "Введи описание (на английском работает лучше):",
            placeholder="например: a cat sitting on a sofa",
        )
        if query:
            with st.spinner("Ищу..."):
                results = searcher.search_by_text(query, top_k=top_k)
            display_results(results, min_score)

    with tab_image:
        uploaded = st.file_uploader(
            "Загрузи картинку для поиска похожих:",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )
        if uploaded:
            query_image = Image.open(uploaded).convert("RGB")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(query_image, caption="Твой запрос", width=200)
            with st.spinner("Ищу похожие..."):
                results = searcher.search_by_image(query_image, top_k=top_k)
            display_results(results, min_score)


if __name__ == "__main__":
    main()
