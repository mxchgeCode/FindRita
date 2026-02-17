import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO


def process_images():
    """Сканирует каталог dataset, обрабатывает все JPG изображения и сохраняет результаты."""
    
    # Пути к каталогам
    dataset_dir = Path("dataset")
    result_dir = Path("result")
    
    # Создаём каталог result, если он не существует
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Каталог result готов: {result_dir.absolute()}")
    
    # Очищаем каталог result перед началом
    for f in result_dir.glob("*"):
        if f.is_file():
            f.unlink()
    print(f"🗑️  Каталог result очищен")
    
    # Удаляем старую папку runs если есть
    if Path("runs").exists():
        try:
            shutil.rmtree("runs")
            print(f"🗑️  Старая папка runs удалена")
        except Exception as e:
            print(f"⚠️  Не удалось удалить runs: {e}")
    
    # Проверяем существование каталога dataset
    if not dataset_dir.exists():
        print(f"❌ Ошибка: каталог dataset не найден: {dataset_dir}")
        sys.exit(1)
    
    # Находим все файлы с расширением .jpg
    jpg_files = list(dataset_dir.glob("*.jpg"))
    
    if not jpg_files:
        print("⚠️  В каталоге dataset не найдены файлы с расширением .jpg")
        return
    
    print(f"🔍 Найдено {len(jpg_files)} изображений для обработки")
    print("-" * 50)
    
    # Загружаем YOLO модель
    print("🐱 Загружаю YOLOv11n...")
    try:
        model = YOLO("yolo11n.pt")
        print("✅ Модель загружена успешно\n")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        sys.exit(1)
    
    # Счётчики для статистики
    success_count = 0
    error_count = 0
    
    # Обрабатываем каждое изображение
    for image_path in sorted(jpg_files):
        filename = image_path.name
        
        try:
            print(f"▶️  Обрабатываю: {filename}")
            
            # Проверяем доступность файла
            if not os.access(image_path, os.R_OK):
                print(f"   ❌ Ошибка доступа: файл недоступен для чтения")
                error_count += 1
                continue
            
            # Очищаем временную папку runs перед каждым изображением
            if Path("runs").exists():
                shutil.rmtree("runs", ignore_errors=True)
            
            # Обрабатываем изображение
            results = model(
                source=str(image_path),
                conf=0.25,
                save=True,
                project="runs/detect",
                name="temp",
                exist_ok=True
            )
            
            # Получаем путь сохранения из результата
            if results and len(results) > 0:
                save_dir = results[0].save_dir
                
                if save_dir:
                    # Ищем все файлы в директории сохранения
                    save_path = Path(save_dir)
                    if save_path.exists():
                        for f in save_path.glob("*.jpg"):
                            dest_path = result_dir / f.name
                            shutil.copy2(f, dest_path)
                            print(f"   ✅ Сохранено: {f.name}")
                    else:
                        print(f"   ⚠️ Директория не существует: {save_path}")
                
                print(f"   ✅ Успешно обработано: {filename}")
                success_count += 1
            else:
                print(f"   ⚠️  Результат пустой для {filename}")
                success_count += 1
            
        except PermissionError as e:
            print(f"   ❌ Ошибка доступа к файлу {filename}: {e}")
            error_count += 1
            
        except OSError as e:
            print(f"   ❌ Ошибка OS при обработке {filename}: {e}")
            error_count += 1
            
        except Exception as e:
            print(f"   ❌ Неожиданная ошибка при обработке {filename}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    # Удаляем временную папку runs в конце
    if Path("runs").exists():
        shutil.rmtree("runs", ignore_errors=True)
        print(f"🗑️  Папка runs удалена")
    
    # Выводим итоговую статистику
    print("-" * 50)
    print(f"📊 Обработка завершена:")
    print(f"   ✅ Успешно: {success_count}")
    if error_count > 0:
        print(f"   ❌ Ошибок: {error_count}")
    print(f"   📁 Результаты сохранены в: {result_dir.absolute()}")


if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
