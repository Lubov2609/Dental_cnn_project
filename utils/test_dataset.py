import os
import sys

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dental_dataset import DentalDataset

# Корректный путь к CSV
csv_file = os.path.join("data", "splits", "train.csv")
img_dir = os.path.join("data", "processed")  # или "data/raw", если используете оригинальные изображения

# Создаём экземпляр датасета
ds = DentalDataset(csv_file=csv_file, img_dir=img_dir)

# Проверяем длину датасета
print(f"Размер датасета: {len(ds)}")

# Получаем первый элемент
img, label = ds[0]

# Выводим информацию об изображении и метке
print(f"Форма изображения: {img.size}")  # Для PIL Image
print(f"Тип изображения: {type(img)}")
print(f"Метки: {label}")