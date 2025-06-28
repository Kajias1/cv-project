import cv2
import os
from pathlib import Path
from tqdm import tqdm

RAW_VIDEO_DIR = Path("data/raw")
OUTPUT_FRAMES_DIR = Path("data/frames")
SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def extract_frames_from_video(video_path: Path, output_dir: Path, every_nth: int = 10, scale: float = 0.5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Ошибка открытия: {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(desc=f"[{video_path.name}]", total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_nth == 0:
            if scale != 1.0:
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            frame_filename = output_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_idx += 1

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

def main():
    print(f"[i] Поиск видео в: {RAW_VIDEO_DIR}")
    video_files = [f for f in RAW_VIDEO_DIR.glob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not video_files:
        print("[!] Видео не найдено в data/raw/")
        return
    
    print(f"[✓] Найдено видео файлов: {len(video_files)}")

    print(f"[i] Извлечение кадров...")
    for video_file in video_files:
        video_stem = video_file.stem
        output_path = OUTPUT_FRAMES_DIR / video_stem
        extract_frames_from_video(video_file, output_path)

    print("[✓] Извлечение кадров завершено.")


if __name__ == "__main__":
    main()
