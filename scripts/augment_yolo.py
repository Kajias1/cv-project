import os
from pathlib import Path
import cv2
import albumentations as A

BASE_DIR = Path(__file__).resolve().parent.parent


def read_yolo_annotations(txt_path):
    """
    Читает YOLO-аннотации из файла и возвращает список кортежей
    (class_id, x_center, y_center, width, height), все значения float.
    """
    ann = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:5]))
            ann.append((cls, *coords))
    return ann

def write_yolo_annotations(annotations, out_txt_path):
    """
    annotations: список кортежей (class_id, x_center, y_center, w, h)
    """
    with open(out_txt_path, 'w') as f:
        for cls, x, y, w, h in annotations:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def make_transform():
    # Пример пайплайна: яркость/контраст, поворот, отзеркаливание
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.RandomScale(scale_limit=0.2, p=0.5),
    ], 
    bbox_params=A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_visibility=0.3
    ))

def augment_dataset(
    base_img_dir: Path,
    base_txt_dir: Path,
    train_list: Path,
    out_img_dir: Path,
    out_txt_dir: Path,
    n_augs: int = 3
):
    transform = make_transform()
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_txt_dir.mkdir(parents=True, exist_ok=True)

    with open(train_list, 'r') as list_file:
        lines = [l.strip() for l in list_file if l.strip()]

    for rel_path in lines:
        img_name = Path(rel_path).name          # 000000.jpg
        stem = img_name.rsplit('.',1)[0]        # 000000
        img_path = base_img_dir / img_name
        txt_path = base_txt_dir / f"{stem}.txt"

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        yolo_anns = read_yolo_annotations(txt_path)
        # распакуем для Albumentations
        bboxes = [ann[1:] for ann in yolo_anns]
        class_ids = [ann[0] for ann in yolo_anns]

        for i in range(n_augs):
            augmented = transform(
                image=image, 
                bboxes=bboxes, 
                class_labels=class_ids
            )
            aug_img = augmented['image']
            aug_boxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            out_img_name = f"{stem}_aug{i}.jpg"
            out_txt_name = f"{stem}_aug{i}.txt"
            cv2.imwrite(str(out_img_dir / out_img_name), aug_img)

            # сохраняем нормализованные аннотации YOLO
            write_yolo_annotations(
                zip(aug_labels, *zip(*aug_boxes)), 
                out_txt_dir / out_txt_name
            )

    print(f"[✓] Аугментация завершена, сохранено в {out_img_dir}")

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent   # корень проекта

    frames_root = BASE_DIR / "data" / "frames"
    ann_root    = BASE_DIR / "data" / "annotations"
    aug_root    = BASE_DIR / "data" / "augmented"

    if not frames_root.exists():
        print(f"[ERROR] Нет папки с кадрами: {frames_root}")
        sys.exit(1)

    # Проходим по каждой подпапке в data/frames/
    for folder in sorted(frames_root.iterdir()):
        if not folder.is_dir():
            continue

        vid_id = folder.name
        print(f"\n[i] === Обрабатываем набор «{vid_id}» ===")

        # 1) картинки
        base_img_dir = frames_root / vid_id

        # 2) аннотации
        base_txt_dir = ann_root / vid_id / "obj_Train_data"
        train_list   = ann_root / vid_id / "train.txt"

        # 3) куда сохранять
        out_imgs = aug_root / vid_id / "images"
        out_txts = aug_root / vid_id / "labels"

        # Диагностика
        print(f"[i]  Изображения:       {base_img_dir}")
        print(f"[i]  YOLO-теги:         {base_txt_dir}")
        print(f"[i]  train.txt:          {train_list}")
        print(f"[i]  Выход (картинки):   {out_imgs}")
        print(f"[i]  Выход (аннотации):  {out_txts}")

        # Проверки
        missing = False
        for p in (base_img_dir, base_txt_dir, train_list):
            if not p.exists():
                print(f"[ERROR] Не найдено: {p}")
                missing = True
        if missing:
            print(f"[WARN] Пропускаем набор «{vid_id}» из-за отсутствующих файлов.")
            continue

        # Аугментация
        augment_dataset(
            base_img_dir=base_img_dir,
            base_txt_dir=base_txt_dir,
            train_list=train_list,
            out_img_dir=out_imgs,
            out_txt_dir=out_txts,
            n_augs=5
        )

    print("\n[i] Все наборы обработаны.")


if __name__ == "__main__":
    main()
