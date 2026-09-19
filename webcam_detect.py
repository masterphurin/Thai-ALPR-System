import argparse
import os
from collections import deque

import cv2
import easyocr
from ultralytics import YOLO


PLATE_DIR = 'plates'
RESULTS_FILE = 'results.txt'
MODEL_PATH = 'license_plate_detector.pt'
MIN_PLATE_CONFIDENCE = 0.50


def intersection_over_union(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return overlap / float(first_area + second_area - overlap)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def read_best_text(crop, reader):
    if crop.size == 0:
        return '', 0.0

    scale = max(4, min(8, 1200 // max(1, crop.shape[1])))
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharpened = cv2.addWeighted(contrast, 2.0, cv2.GaussianBlur(contrast, (0, 0), 3), -1.0, 0)
    variants = [enlarged, gray, contrast, sharpened,
                cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 7)]
    candidates = []
    for variant in variants:
        for _, text, confidence in reader.readtext(
                variant, detail=1, paragraph=False, mag_ratio=1.5,
                min_size=5, text_threshold=0.3, low_text=0.2, link_threshold=0.2):
            cleaned = ''.join(text.split()).strip()
            if cleaned and confidence >= 0.10:
                score = float(confidence) * (1.0 + min(len(cleaned), 8) * 0.08)
                candidates.append((cleaned, float(confidence), score))
    best = max(candidates, key=lambda item: item[2], default=('', 0.0, 0.0))
    return best[0], best[1]


def make_panel(frame, captures, panel_width=300):
    panel = cv2.resize(frame, (panel_width, frame.shape[0]))
    cv2.rectangle(panel, (0, 0), (panel_width, 42), (30, 30, 30), -1)
    cv2.putText(panel, 'PLATES', (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    y = 55
    for image, label in reversed(captures):
        if y + 105 > panel.shape[0]:
            break
        thumbnail = cv2.resize(image, (panel_width - 20, 78))
        panel[y:y + 78, 10:panel_width - 10] = thumbnail
        cv2.putText(panel, label[:32], (10, y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        y += 112
    return panel


def process_video(video_path):
    os.makedirs(PLATE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f'ไม่สามารถเปิดไฟล์วิดีโอได้: {video_path}')

    print('กำลังโหลดโมเดล YOLO และ EasyOCR...')
    plate_detector = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['th', 'en'], gpu=False)
    captures = deque(maxlen=8)
    tracks = []
    frame_number = 0
    capture_number = 0
    frame_gap = max(1, int(cap.get(cv2.CAP_PROP_FPS) * 0.7))

    with open(RESULTS_FILE, 'w', encoding='utf-8') as results_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            height, width = frame.shape[:2]
            detections = []
            result = plate_detector(frame, verbose=False, conf=MIN_PLATE_CONFIDENCE, iou=0.45)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), width, height)
                confidence = float(box.conf[0])
                if confidence >= MIN_PLATE_CONFIDENCE and x2 > x1 and y2 > y1:
                    detections.append(((x1, y1, x2, y2), confidence))

            for box, confidence in detections:
                matching_track = next((track for track in tracks if intersection_over_union(box, track['box']) > 0.20), None)
                if matching_track is None:
                    matching_track = {
                        'box': box,
                        'best_score': 0,
                        'first_frame': frame_number,
                        'last_frame': frame_number,
                        'captured': False,
                    }
                    tracks.append(matching_track)
                matching_track['box'] = box
                crop = frame[box[1]:box[3], box[0]:box[2]]
                sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                quality = confidence * (1.0 + min(sharpness / 100.0, 3.0))
                if quality > matching_track['best_score']:
                    matching_track['best_score'] = quality
                    matching_track['best_crop'] = crop.copy()
                if frame_number - matching_track['last_frame'] > frame_gap:
                    matching_track['captured'] = False
                matching_track['last_frame'] = frame_number

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'PLATE {confidence:.2f}', (box[0], max(24, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                if not matching_track['captured'] and frame_number - matching_track['first_frame'] >= 3:
                    capture_number += 1
                    crop = matching_track.get('best_crop', frame[box[1]:box[3], box[0]:box[2]])
                    filename = os.path.join(PLATE_DIR, f'plate_{capture_number:04d}.jpg')
                    cv2.imwrite(filename, crop)
                    text, text_confidence = read_best_text(crop, reader)
                    label = text or 'อ่านไม่ออก'
                    results_file.write(f'ภาพที่ {capture_number} (ไฟล์ {filename}): {label} confidence={text_confidence:.3f}\n')
                    results_file.flush()
                    captures.append((crop, f'{capture_number}: {label}'))
                    matching_track['captured'] = True
                    print(f'แคป {filename} -> {label}')

            tracks = [track for track in tracks if frame_number - track['last_frame'] <= frame_gap * 3]
            display = cv2.hconcat([make_panel(frame, captures), frame])
            cv2.imshow('Video ALPR - plates', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ตรวจจับและอ่านป้ายทะเบียนจากไฟล์วิดีโอ')
    parser.add_argument('video', nargs='?', default=os.getenv('VIDEO_PATH'), help='พาธไฟล์วิดีโอ')
    args = parser.parse_args()
    if not args.video:
        raise SystemExit('กรุณาระบุไฟล์วิดีโอ เช่น python webcam_detect.py video.mp4')
    process_video(args.video)