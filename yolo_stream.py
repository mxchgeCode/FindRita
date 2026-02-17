import sys
import cv2
import time
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


def check_cv2_gui_support():
    """Проверяет, поддерживает ли OpenCV GUI функции."""
    print("🔍 Проверяю поддержку GUI в OpenCV...")
    try:
        # Пробуем создать тестовое окно
        test_window = "test_cv2_gui"
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("✅ GUI поддерживается!")
        return True
    except Exception as e:
        print(f"❌ GUI НЕ поддерживается: {e}")
        return False


def run_webcam_stream(camera_index=0, show_fps=True, output_file=None, window_name="YOLO Cat Detection"):
    """
    Запускает обнаружение кошек с вебкамеры в реальном времени.
    
    Параметры:
        camera_index: индекс камеры (0 - первая камера)
        show_fps: показывать FPS
        output_file: путь для сохранения видео (если None - показывать на экране)
        window_name: название окна
    
    Возвращает:
        None
    """
    
    # Проверяем поддержку GUI
    gui_supported = check_cv2_gui_support()
    print(f"📦 OpenCV версия: {cv2.__version__}")
    
    # Загружаем YOLO модель
    print("🐱 Загружаю YOLOv11n...")
    try:
        model = YOLO("yolo11n.pt")
        print("✅ Модель загружена успешно\n")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        sys.exit(1)
    
    # Открываем вебкамеру
    print(f"📷 Открываю камеру {camera_index}...")
    
    # Пробуем разные backend для захвата видео
    backends = [
        (cv2.CAP_ANY, "CAP_ANY"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_VFW, "VFW"),
    ]
    
    cap = None
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(camera_index + backend)
            if cap.isOpened():
                # Проверяем, можем ли получить кадр
                ret, test_frame = cap.read()
                if ret:
                    print(f"✅ Камера открыта с backend: {name}")
                    # Возвращаем кадр в буфер
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                else:
                    cap.release()
                    cap = None
            else:
                cap = None
        except Exception as e:
            print(f"⚠️ Backend {name} не работает: {e}")
    
    if cap is None or not cap.isOpened():
        # Последняя попытка - без явного backend
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Ошибка: не удалось открыть камеру {camera_index}")
        sys.exit(1)
    
    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"✅ Камера открыта: {width}x{height} @ {fps} fps")
    
    # Настраиваем вывод
    writer = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"📹 Видео будет сохранено: {output_path.absolute()}")
        print("-" * 50)
    elif gui_supported:
        print("-" * 50)
        print("🎯 Нажмите 'q' или 'ESC' для выхода")
        print("-" * 50)
    else:
        print("⚠️  Внимание: OpenCV без поддержки GUI!")
        print("💡 Используйте --output для сохранения видео в файл")
        print("-" * 50)
    
    # Параметры фильтрации для устранения ложных срабатываний
    FILTER_MIN_CONF = 0.85    # Минимальная уверенность
    FILTER_MIN_SIZE = 50       # Минимальный размер (пикселей)
    FILTER_MIN_ASPECT = 0.5    # Минимальное соотношение сторон (w/h)
    FILTER_MAX_ASPECT = 2.0    # Максимальное соотношение сторон (w/h)
    
    # Переменные для FPS
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Трекинг: запоминаем последнюю обнаруженную позицию кошки
    last_valid_box = None
    last_valid_conf = 0.0
    frames_since_last_detection = 0
    MAX_FRAMES_WITHOUT_DETECTION = 30  # ~1 секунда при 30fps
    
    # Флаг для отображения "CAT DETECTED!"
    cat_detected = False
    detection_time = 0
    
    # Счётчик кадров
    frame_count = 0
    cats_total = 0
    
    # Основной цикл
    running = True
    while running and cap.isOpened():
        # Читаем кадр
        ret, frame = cap.read()
        if not ret:
            print("❌ Ошибка чтения кадра")
            break
        
        frame_count += 1
        
        # Обрабатываем кадр через YOLO
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
                    cats_total += 1
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
                        
                        # Рисуем оранжевую рамку
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 4)
                        
                        # Добавляем подпись
                        label = f"Cat: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Трекинг: если нет валидной детекции, используем последнюю известную позицию
        if not has_valid_detection and last_valid_box is not None:
            frames_since_last_detection += 1
            if frames_since_last_detection <= MAX_FRAMES_WITHOUT_DETECTION:
                # Рисуем рамку (более тонкую для трекинга)
                x1, y1, x2, y2 = map(int, last_valid_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                label = f"Cat (track): {last_valid_conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Обновляем статус обнаружения
        if has_valid_detection:
            cat_detected = True
            detection_time = time.time()
        elif time.time() - detection_time > 2.0:
            cat_detected = False
        
        # Рисуем индикатор обнаружения
        if cat_detected:
            # Красный индикатор "CAT DETECTED!"
            cv2.putText(frame, "🐱 CAT DETECTED!", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.circle(frame, (width - 50, 50), 20, (0, 0, 255), -1)
        
        # Вычисляем FPS
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()
        
        # Показываем FPS
        if show_fps:
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Вывод: в файл или на экран
        if writer is not None:
            writer.write(frame)
        elif gui_supported:
            try:
                # Показываем кадр
                cv2.imshow(window_name, frame)
                # cv2.waitKey(1) обрабатывается ниже
            except Exception as e:
                print(f"⚠️ Ошибка отображения: {e}")
                print("💡 Переключаюсь на режим записи в файл...")
                output_path = Path("result/stream_output.mp4")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                gui_supported = False
        
        # Проверяем нажатие клавиш (только если GUI доступен)
        if gui_supported:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' или ESC
                print("👋 Выход по запросу пользователя")
                running = False
        
        # Также выходим по Ctrl+C в консоли (проверяем каждый 100 кадр)
        if frame_count % 100 == 0:
            print(f"⏳ Обработано кадров: {frame_count}, кошек обнаружено: {cats_total}")
    
    # Освобождаем ресурсы
    cap.release()
    if writer is not None:
        writer.release()
    
    if gui_supported:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    
    print(f"\n📊 Статистика:")
    print(f"   Всего кадров: {frame_count}")
    print(f"   Кошек обнаружено: {cats_total}")
    print("✅ Ресурсы освобождены")


def main():
    """Точка входа для запуска потокового обнаружения."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Обнаружение кошек с вебкамеры в реальном времени"
    )
    parser.add_argument(
        "-c", "--camera", 
        type=int, 
        default=0, 
        help="Индекс камеры (по умолчанию: 0)"
    )
    parser.add_argument(
        "--no-fps", 
        action="store_true", 
        help="Не показывать FPS"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default=None,
        help="Путь для сохранения видео (по умолчанию: показывать на экране)"
    )
    parser.add_argument(
        "--window-name", 
        type=str, 
        default="YOLO Cat Detection", 
        help="Название окна"
    )
    
    args = parser.parse_args()
    
    try:
        run_webcam_stream(
            camera_index=args.camera,
            show_fps=not args.no_fps,
            output_file=args.output,
            window_name=args.window_name
        )
    except KeyboardInterrupt:
        print("\n⚠️ Прервано пользователем")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
