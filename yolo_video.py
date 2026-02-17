import os
import sys
import shutil
import time
from pathlib import Path
from ultralytics import YOLO


def process_videos():
    """Сканирует каталог dataset, обрабатывает все видео файлы и сохраняет результаты."""
    
    # Пути к каталогам
    dataset_dir = Path("dataset")
    result_dir = Path("result\\video")
    
    # Создаём каталог result, если он не существует
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Каталог result готов: {result_dir.absolute()}")
    
    # Очищаем каталог result перед началом
    for f in result_dir.glob("*"):
        if f.is_file():
            f.unlink()
    print(f"🗑️  Каталог result очищен")
    
    # Проверяем существование каталога dataset
    if not dataset_dir.exists():
        print(f"❌ Ошибка: каталог dataset не найден: {dataset_dir}")
        sys.exit(1)
    
    # Находим все видео файлы (MOV, MP4, AVI) - уникальные
    video_extensions = [".mov", ".mp4", ".avi", ".mkv"]
    video_files = []
    seen_names = set()
    
    for ext in video_extensions:
        for video_file in dataset_dir.glob(f"*{ext}"):
            # Используем lower для уникальности
            name_lower = video_file.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                video_files.append(video_file)
    
    if not video_files:
        print("⚠️  В каталоге dataset не найдены видео файлы")
        return
    
    print(f"🔍 Найдено {len(video_files)} видео файлов для обработки")
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
    
    # Класс "cat" в COCO - номер 15
    CAT_CLASS = 15
    
    # Обрабатываем каждое видео
    for video_path in sorted(video_files):
        filename = video_path.name
        
        try:
            print(f"▶️  Обрабатываю: {filename}")
            
            # Проверяем доступность файла
            if not os.access(video_path, os.R_OK):
                print(f"   ❌ Ошибка доступа: файл недоступен для чтения")
                error_count += 1
                continue
            
            # Удаляем папку runs перед обработкой
            if Path("runs").exists():
                try:
                    shutil.rmtree("runs", ignore_errors=True)
                except:
                    pass
            
            time.sleep(0.5)  # Даём время на удаление
            
            # Обрабатываем видео с фильтрацией только кошек (класс 15)
            results = model(
                source=str(video_path),
                conf=0.75,
                save=True,
                project="runs/detect",
                name="temp",
                exist_ok=True,
                classes=[CAT_CLASS],  # Только кошки
                verbose=False  # Отключаем покадровый вывод
            )
            
            # Получаем путь сохранения
            if results and len(results) > 0:
                save_dir = results[0].save_dir
                
                if save_dir:
                    save_path = Path(save_dir)
                    if save_path.exists():
                        # Ищем AVI файл (YOLO сохраняет видео как AVI)
                        avi_files = list(save_path.glob("*.avi"))
                        if avi_files:
                            for avi_file in avi_files:
                                # Меняем расширение на mp4
                                mp4_name = avi_file.stem + ".mp4"
                                dest_path = result_dir / mp4_name
                                # Копируем файл
                                shutil.copy2(avi_file, dest_path)
                                print(f"   ✅ Сохранено: {mp4_name}")
                        else:
                            print(f"   ⚠️  Не найден AVI файл в: {save_path}")
                    else:
                        print(f"   ⚠️ Директория не существует: {save_path}")
            
            print(f"   ✅ Успешно обработано: {filename}")
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
    time.sleep(1)  # Даём время на завершение записи
    
    if Path("runs").exists():
        for _ in range(3):  # Пробуем 3 раза
            try:
                shutil.rmtree("runs", ignore_errors=True)
                if not Path("runs").exists():
                    print(f"🗑️  Папка runs удалена")
                    break
            except:
                time.sleep(0.5)
    
    # Выводим итоговую статистику
    print("-" * 50)
    print(f"📊 Обработка завершена:")
    print(f"   ✅ Успешно: {success_count}")
    if error_count > 0:
        print(f"   ❌ Ошибок: {error_count}")
    print(f"   📁 Результаты сохранены в: {result_dir.absolute()}")


if __name__ == "__main__":
    try:
        process_videos()
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
