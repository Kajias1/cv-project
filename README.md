# CV Project

Набор скриптов и конфигураций для полного пайплайна по распознаванию объектов (блюд) в видеокадрах:

1. **Извлечение кадров** из видеозаписей  
2. **Аннотация** (YOLO-формат)  
3. **Аугментация** данных с сохранением .jpg + .txt  
4. **Разбиение** `train` / `val` / `test`  
5. **Обучение** модели YOLOv8/v11 на полученном датасете  
6. **Оценка** качества (mAP, precision, recall, F1)

---

## 📁 Структура проекта

```text
cv_project/
├── data/  
│   ├── raw/                    # сырьевые видео  
│   ├── frames/                 # извлечённые кадры  
│   │   └── <id>/  (000000.jpg + .txt/.xml)  
│   ├── annotations/  
│   │   └── <id>/  
│   │       ├── obj_Train_data/ # исходные JPG + YOLO .txt  
│   │       └── train.txt       # список всех JPG  
│   └── augmented/              # результаты аугментации  
├── dataset/                    # финальный датасет  
│   ├── train/  
│   │   ├── images/  
│   │   └── labels/  
│   ├── val/  
│   │   ├── images/  
│   │   └── labels/  
│   └── test/  
│       ├── images/  
│       └── labels/  
├── runs/                       # логи и результаты обучения  
├── scripts/  
│   ├── extract_frames.py       # извлечение кадров  
│   ├── augment_yolo.py         # аугментация YOLO-датасета  
│   └── split_to_dataset.py     # разбиение на train/val/test  
├── venv/                       # виртуальное окружение (не коммитить)  
├── data.yaml                   # конфиг датасета для YOLO  
├── requirements.txt            # все Python-зависимости  
├── yolov8n.pt                  # (или yolov11n.pt) веса модели  
└── README.md
