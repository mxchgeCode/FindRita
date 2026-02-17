import os
import sys
import shutil
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def filter_cat_detections(boxes, min_conf=0.85, min_size=50, min_aspect=0.5, max_aspect=2.0):
    """
    Фильтрует обнаружения кошек, убирая ложные срабатывания (игрушки и т.д.)
    
    Параметры:
        min_conf: минимальная уверенность (по умолчанию 0.85)
        min_size: минимальная ширина/высота в пикселях (по умолчанию 50)
        min_aspect: минимальное соотношение сторон (w/h)
        max_aspect: максимальное соотношение сторон (w/h)
    
    Возвращает:
        Отфильтрованный список боксов
    """
    filtered = []
    
    for box in boxes:
        cls = int(box.cls[0]) if hasattr(box, 'cls') else int(box.cls)
        conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box.conf)
        
        # Проверяем класс (cat = 15 в COCO)
        if cls != 15:
            continue
        
        # Получаем координаты
        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0]
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        
        # Проверяем размер
        if w < min_size or h < min_size:
            continue
        
        # Проверяем соотношение сторон
        aspect_ratio = w / h
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue
        
        # Проверяем уверенность
        if conf < min_conf:
            continue
        
        filtered.append(box)
    
    return filtered


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
    
    # Параметры фильтрации для устранения ложных срабатываний
    FILTER_MIN_CONF = 0.85    # Минимальная уверенность
    FILTER_MIN_SIZE = 50       # Минимальный размер (пикселей)
    FILTER_MIN_ASPECT = 0.5    # Минимальное соотношение сторон (w/h)
    FILTER_MAX_ASPECT = 2.0    # Максимальное соотношение сторон (w/h)
    
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
            
            # Открываем видео для покадровой обработки
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"   ❌ Ошибка: не удалось открыть видео")
                error_count += 1
                continue
            
            # Получаем параметры видео
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Создаём VideoWriter для сохранения результата
            output_path = result_dir / f"{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_idx = 0
            cats_found = 0
            frames_processed = 0
            
            # Трекинг: запоминаем последнюю обнаруженную позицию кошки
            last_valid_box = None
            last_valid_conf = 0.0
            frames_since_last_detection = 0
            MAX_FRAMES_WITHOUT_DETECTION = 60  # Увеличили до 60 кадров (~2 секунды при 30fps)
            
            print(f"   📹 Видео: {width}x{height}, {fps} fps, {total_frames} кадров")
            
            # Обрабатываем каждый кадр
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Обрабатываем каждый кадр
                results = model(frame, conf=0.75, verbose=False)
                
                # Применяем фильтры для устранения ложных срабатываний
                has_valid_detection = False
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        # Фильтруем обнаружения
                        filtered_boxes = filter_cat_detections(
                            boxes,
                            min_conf=FILTER_MIN_CONF,
                            min_size=FILTER_MIN_SIZE,
                            min_aspect=FILTER_MIN_ASPECT,
                            max_aspect=FILTER_MAX_ASPECT
                        )
                        
                        if len(filtered_boxes) > 0:
                            cats_found += 1
                            has_valid_detection = True
                            frames_since_last_detection = 0
                            
                            # Берём первое обнаружение (самое уверенное)
                            box = filtered_boxes[0]
                            last_valid_box = box.xyxy[0].cpu().numpy()
                            last_valid_conf = float(box.conf[0])
                            
                            # Рисуем боксы на кадре
                            for box in filtered_boxes:
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                
                                x1, y1, x2, y2 = map(int, xyxy)
                                
                                # Рисуем одну толстую оранжевую рамку
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 8)
                                
                                # Добавляем подпись
                                label = f"Cat: {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                
                # Трекинг: если нет валидной детекции, используем последнюю известную позицию
                if not has_valid_detection and last_valid_box is not None:
                    frames_since_last_detection += 1
                    if frames_since_last_detection <= MAX_FRAMES_WITHOUT_DETECTION:
                        # Рисуем одну толстую рамку
                        x1, y1, x2, y2 = map(int, last_valid_box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 8)
                        label = f"Cat: {last_valid_conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                
                # Записываем кадр в выходное видео
                out.write(frame)
                frames_processed += 1
                
                # Показываем прогресс
                if frame_idx % 30 == 0:
                    print(f"   ⏳ Обработано кадров: {frame_idx}/{total_frames}")
            
            # Освобождаем ресурсы
            cap.release()
            out.release()
            
            print(f"   ✅ Найдено кошек в {cats_found} кадрах")
            print(f"   ✅ Сохранено: {output_path.name}")
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
