import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def get_transforms():
    # ИСПРАВЛЕННАЯ ВЕРСИЯ: используем квадратные скобки вместо фигурных
    train_transform = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.ToTensor()
    ])
    return train_transform, val_transform

def tensor_to_pil(tensor):
    """Конвертирует тензор PyTorch обратно в PIL Image для визуализации"""
    # Убираем нормализацию: умножаем на std и прибавляем mean
    # Для простоты используем стандартные значения
    tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    tensor = tensor.numpy()
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def test_augmentation(image_path, num_augmentations=5):
    """
    Тестирует аугментацию на одном изображении
    Args:
        image_path: путь к тестовому изображению
        num_augmentations: количество раз применить аугментацию
    """
    # Загружаем изображение
    original_image = Image.open(image_path).convert('RGB')
    print(f"Исходное изображение: {original_image.size}")

    # Получаем преобразования
    train_transform, val_transform = get_transforms()

    # Применяем аугментацию несколько раз
    augmented_images = []
    for i in range(num_augmentations):
        # Создаём копию оригинального изображения
        augmented = train_transform(original_image)
        augmented_images.append(augmented)

    # Визуализируем результаты
    fig, axes = plt.subplots(2, (num_augmentations + 1) // 2 + 1,
                         figsize=(15, 8))
    axes = axes.ravel()

    # Показываем оригинальное изображение
    axes[0].imshow(original_image)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Показываем аугментированные изображения
    for i, aug_tensor in enumerate(augmented_images):
        aug_pil = tensor_to_pil(aug_tensor)
        axes[i + 1].imshow(aug_pil)
        axes[i + 1].set_title(f"Аугментация {i+1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

    # Дополнительно: проверяем, что изображения действительно разные
    print("\nПроверка уникальности аугментированных изображений:")
    unique_hashes = set()
    for i, img_tensor in enumerate(augmented_images):
        # Конвертируем в numpy и вычисляем хеш
        img_array = img_tensor.numpy().tobytes()
        img_hash = hash(img_array)
        unique_hashes.add(img_hash)
        print(f"Аугментация {i+1} хеш: {img_hash % 10000}")  # показываем последние 4 цифры

    print(f"\nВсего уникальных изображений: {len(unique_hashes)} из {num_augmentations}")
    if len(unique_hashes) == num_augmentations:
        print("✅ Все аугментированные изображения уникальны — аугментация работает!")
    else:
        print("❌ Некоторые аугментированные изображения идентичны — проверьте параметры аугментации")

# Запуск теста
if __name__ == "__main__":
    # Укажите путь к вашему тестовому изображению
    # Пример: "data/test_image.jpg"
    test_image_path = "data/processed/32.jpg"  # ЗАМЕНИТЕ НА ВАШ ПУТЬ

    if os.path.exists(test_image_path):
        test_augmentation(test_image_path, num_augmentations=6)
    else:
        print(f"Ошибка: файл {test_image_path} не найден!")
        print("Пожалуйста, укажите корректный путь к тестовому изображению.")
