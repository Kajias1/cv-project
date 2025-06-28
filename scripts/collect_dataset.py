import sys
import shutil
from pathlib import Path

def collect_all(
    aug_root: Path,
    out_root: Path
):
    """
    aug_root: папка data/augmented/
    out_root: папка dataset/  (создаст внутри images/ и labels/)
    """
    images_out = out_root / "images"
    labels_out = out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"[i] Идёт процесс сборки всех изображений и тэгов воедино")

    for subset in sorted(aug_root.iterdir()):
        if not subset.is_dir():
            continue
        imgs_dir = subset / "images"
        lbls_dir = subset / "labels"
        if not imgs_dir.exists() or not lbls_dir.exists():
            print(f"[WARN] Пропускаем {subset.name}: нет папки images/labels")
            continue

        for img_path in imgs_dir.glob("*.jpg"):
            stem = img_path.stem
            lbl_path = lbls_dir / f"{stem}.txt"
            if not lbl_path.exists():
                print(f"[WARN] Для {img_path.name} нет метки, пропускаем")
                continue

            dst_img = images_out / img_path.name
            dst_lbl = labels_out / lbl_path.name
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

    print(f"\n[i] Сборка датасета завершена. Всего изображений: {len(list(images_out.glob('*.jpg')))}")
    print(f"[i] Папки созданы в: {out_root}")

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    aug_root = BASE_DIR / "data" / "augmented"
    out_root = BASE_DIR / "dataset"

    if not aug_root.exists():
        print(f"[ERROR] Не найдена папка с аугментированными данными: {aug_root}")
        sys.exit(1)

    collect_all(aug_root, out_root)

if __name__ == "__main__":
    main()
