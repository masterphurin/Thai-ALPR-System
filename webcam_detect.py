import argparse
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


PLATE_DIR = 'plates'
RESULTS_FILE = 'results.txt'
MODEL_PATH = 'license_plate_detector.pt'
MIN_PLATE_CONFIDENCE = 0.50
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_DETECTION_INTERVAL = 1
MIN_TRACK_FRAMES = 6
MAX_CROP_CANDIDATES = 5
OCR_CROP_CANDIDATES = 3
OCR_VARIANT_COUNT = 3
OCR_MIN_CONFIDENCE = 0.08
MIN_ACCEPTED_OCR_CONFIDENCE = 0.10
MIN_ACCEPTED_PLATE_LENGTH = 4
MIN_ACCEPTED_PLATE_DIGITS = 2
PLATE_PADDING_RATIO = 0.10
MIN_CAPTURE_PLATE_WIDTH = 90
SHRINKING_FRAMES_TO_CAPTURE = 2

DIGIT_GUESSES = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'T': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
}


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


def crop_plate(frame, box, padding_ratio=PLATE_PADDING_RATIO):
    """Crop a plate with a small border so tight detector boxes do not cut characters."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(2, int((x2 - x1) * padding_ratio))
    pad_y = max(2, int((y2 - y1) * padding_ratio))
    x1, y1, x2, y2 = clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
    return frame[y1:y2, x1:x2]


def clean_ocr_text(text):
    """Keep Thai/Latin letters and digits while removing OCR punctuation and spaces."""
    text = str(text).upper()
    return ''.join(character for character in text if character.isalnum() or '\u0e00' <= character <= '\u0e7f')


def plate_shape_score(text):
    """Score candidates that resemble a Thai registration number over random short detections."""
    length = len(text)
    digit_count = sum(character.isdigit() for character in text)
    thai_count = sum('\u0e00' <= character <= '\u0e7f' for character in text)
    latin_count = sum('A' <= character <= 'Z' for character in text)

    if length < 4:
        # A one-character high-confidence detection is usually a border or province glyph.
        return 0.12
    score = 1.0 if 4 <= length <= 8 else 0.65
    if digit_count >= 2:
        score += 0.25
    if thai_count >= 1 or latin_count >= 1:
        score += 0.10
    return score


def looks_like_plate(text):
    """Reject confident one-letter OCR hallucinations that are not plate-shaped."""
    digit_count = sum(character.isdigit() for character in text)
    return len(text) >= MIN_ACCEPTED_PLATE_LENGTH and digit_count >= MIN_ACCEPTED_PLATE_DIGITS


def guess_plate_texts(text):
    """Return the OCR text plus cautious guesses for its numeric registration part."""
    cleaned = clean_ocr_text(text)
    if not cleaned:
        return []

    guesses = [(cleaned, 1.0)]
    thai_positions = [
        index for index, character in enumerate(cleaned)
        if '\u0e00' <= character <= '\u0e7f'
    ]
    if thai_positions:
        numeric_start = thai_positions[-1] + 1
        numeric_part = cleaned[numeric_start:]
        if 2 <= len(numeric_part) <= 5:
            guessed_numeric_part = ''.join(
                DIGIT_GUESSES.get(character, character) for character in numeric_part
            )
            mapped_count = sum(character.isdigit() for character in guessed_numeric_part)
            if mapped_count >= 2 and guessed_numeric_part != numeric_part:
                guesses.append((cleaned[:numeric_start] + guessed_numeric_part, 0.90))
                return guesses

    first_digit = next(
        (index for index, character in enumerate(cleaned) if character.isdigit()),
        None,
    )
    if first_digit is not None and first_digit < len(cleaned) - 1:
        numeric_part = cleaned[first_digit:]
        guessed_numeric_part = ''.join(DIGIT_GUESSES.get(character, character) for character in numeric_part)
        if guessed_numeric_part != numeric_part:
            guesses.append((cleaned[:first_digit] + guessed_numeric_part, 0.94))
        return guesses

    for suffix_length in range(2, min(4, len(cleaned)) + 1):
        split = len(cleaned) - suffix_length
        suffix = cleaned[split:]
        mapped_suffix = ''.join(DIGIT_GUESSES.get(character, character) for character in suffix)
        mapped_count = sum(character.isdigit() for character in mapped_suffix)
        if mapped_count >= 2 and mapped_suffix != suffix:
            guesses.append((cleaned[:split] + mapped_suffix, 0.82))
    return guesses


def _ocr_candidates(detections, image_height):
    """Convert EasyOCR boxes to line candidates and ignore the province text line."""
    usable = []
    for bbox, text, confidence in detections:
        cleaned = clean_ocr_text(text)
        if not cleaned or float(confidence) < OCR_MIN_CONFIDENCE:
            continue
        if bbox and isinstance(bbox[0], (int, float)):
            x1, x2, y1, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
        else:
            center_y = sum(point[1] for point in bbox) / max(1, len(bbox))
            center_x = sum(point[0] for point in bbox) / max(1, len(bbox))
        usable.append((center_x, center_y, cleaned, float(confidence)))

    if not usable:
        return []

    # Thai plates normally put the registration number above the province name.
    main_line = [item for item in usable if item[1] <= image_height * 0.72]
    if not main_line:
        main_line = usable
    main_line.sort(key=lambda item: item[0])

    candidates = [(text, confidence) for _, _, text, confidence in main_line]
    if len(main_line) > 1:
        joined = clean_ocr_text(''.join(item[2] for item in main_line))
        joined_confidence = sum(item[3] for item in main_line) / len(main_line)
        if joined:
            candidates.append((joined, joined_confidence))
    return candidates


def choose_text(candidates):
    """Choose a stable result by combining repeated OCR text across variants/frames."""
    grouped = {}
    for text, confidence in candidates:
        for cleaned, guess_confidence in guess_plate_texts(text):
            confidence = float(confidence) * guess_confidence
            shape = plate_shape_score(cleaned)
            weighted_score = confidence * shape
            item = grouped.setdefault(cleaned, {'total': 0.0, 'max_confidence': 0.0, 'count': 0})
            item['total'] += weighted_score
            item['max_confidence'] = max(item['max_confidence'], confidence)
            item['count'] += 1

    if not grouped:
        return '', 0.0

    text, values = max(
        grouped.items(),
        key=lambda item: (item[1]['total'], item[1]['count'], item[1]['max_confidence'], len(item[0])),
    )
    return text, values['max_confidence']


def read_best_text(crop, reader):
    if crop.size == 0:
        return '', 0.0

    scale = max(4, min(10, 1400 // max(1, crop.shape[1])))
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    enlarged = cv2.copyMakeBorder(enlarged, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharpened = cv2.addWeighted(contrast, 1.6, blurred, -0.6, 0)
    thresholded = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants = [gray, sharpened, thresholded][:OCR_VARIANT_COUNT]
    candidates = []
    for variant in variants:
        detections = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            decoder='greedy',
        )
        candidates.extend(_ocr_candidates(detections, variant.shape[0]))
    return choose_text(candidates)


def crop_quality(crop, detector_confidence):
    """Rank crops using confidence, usable size, focus, and exposure."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = float(np.clip(np.log1p(sharpness) / 8.0, 0.0, 1.0))
    exposure_score = 1.0 - min(abs(float(gray.mean()) - 145.0) / 145.0, 1.0)
    size_score = float(np.clip(crop.shape[1] / MIN_CAPTURE_PLATE_WIDTH, 0.0, 1.0))
    return float(detector_confidence) * (0.55 + 0.45 * sharpness_score) * \
        (0.65 + 0.35 * exposure_score) * (0.40 + 0.60 * size_score)


def remember_crop(track, crop, quality):
    if crop.size == 0:
        return
    candidates = track.setdefault('crop_candidates', [])
    candidates.append((quality, crop.copy()))
    candidates.sort(key=lambda item: item[0], reverse=True)
    del candidates[MAX_CROP_CANDIDATES:]
    track['best_crop'] = candidates[0][1]


def read_crop_candidates(track, reader):
    observations = []
    for _, crop in track.get('crop_candidates', [])[:OCR_CROP_CANDIDATES]:
        text, confidence = read_best_text(crop, reader)
        if text:
            observations.append((text, confidence))
    text, confidence = choose_text(observations)
    if confidence < MIN_ACCEPTED_OCR_CONFIDENCE or not looks_like_plate(text):
        return '', confidence
    return text, confidence


def queue_track_capture(track, capture_number, reader, ocr_executor, captures, pending_ocr):
    """Save the best crop and run OCR in the background so the preview stays responsive."""
    if track.get('captured') or track.get('best_crop') is None:
        return capture_number

    capture_number += 1
    crop = track['best_crop']
    filename = os.path.join(PLATE_DIR, f'plate_{capture_number:04d}.jpg')
    if not cv2.imwrite(filename, crop):
        print(f'บันทึกรูปไม่สำเร็จ: {filename}')
        return capture_number - 1

    # Copy the small set of crops because the track is removed after it leaves the frame.
    ocr_track = {
        'crop_candidates': [(quality, candidate.copy()) for quality, candidate in track.get('crop_candidates', [])],
    }
    future = ocr_executor.submit(read_crop_candidates, ocr_track, reader)
    entry = [crop, f'{capture_number}: กำลังอ่าน...']
    captures.append(entry)
    pending_ocr.append({'future': future, 'entry': entry, 'filename': filename, 'number': capture_number})
    track['captured'] = True
    print(f'แคป {filename} -> กำลังอ่าน OCR')
    return capture_number


def collect_ocr_results(pending_ocr, results_file):
    """Publish completed OCR results without blocking the video loop."""
    for job in pending_ocr[:]:
        if not job['future'].done():
            continue
        try:
            text, confidence = job['future'].result()
        except Exception as error:
            text, confidence = '', 0.0
            print(f"OCR ล้มเหลว {job['filename']}: {error}")
        label = text or 'อ่านไม่ออก'
        job['entry'][1] = f"{job['number']}: {label}"
        results_file.write(
            f"ภาพที่ {job['number']} (ไฟล์ {job['filename']}): "
            f"{label} confidence={confidence:.3f}\n"
        )
        results_file.flush()
        print(f"OCR {job['filename']} -> {label} confidence={confidence:.3f}")
        pending_ocr.remove(job)


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


def process_video(
    video_path,
    image_size=DEFAULT_IMAGE_SIZE,
    detection_interval=DEFAULT_DETECTION_INTERVAL,
):
    os.makedirs(PLATE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f'ไม่สามารถเปิดไฟล์วิดีโอได้: {video_path}')

    print('กำลังโหลดโมเดล YOLO และ EasyOCR...')
    plate_detector = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['th', 'en'], gpu=False, verbose=False)
    captures = deque(maxlen=8)
    tracks = []
    pending_ocr = []
    frame_number = 0
    capture_number = 0
    frame_gap = max(1, int(cap.get(cv2.CAP_PROP_FPS) * 0.7))

    with ThreadPoolExecutor(max_workers=1) as ocr_executor, \
            open(RESULTS_FILE, 'w', encoding='utf-8') as results_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            collect_ocr_results(pending_ocr, results_file)
            height, width = frame.shape[:2]
            detections = []
            if frame_number % max(1, detection_interval) == 0:
                result = plate_detector(
                    frame,
                    verbose=False,
                    conf=MIN_PLATE_CONFIDENCE,
                    iou=0.45,
                    imgsz=image_size,
                )[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), width, height)
                    confidence = float(box.conf[0])
                    if confidence >= MIN_PLATE_CONFIDENCE and x2 > x1 and y2 > y1:
                        detections.append(((x1, y1, x2, y2), confidence))

            for box, confidence in detections:
                matching_track = None
                matching_iou = 0.20
                for track in tracks:
                    if frame_number - track['last_frame'] > frame_gap:
                        continue
                    overlap = intersection_over_union(box, track['box'])
                    if overlap > matching_iou:
                        matching_iou = overlap
                        matching_track = track
                if matching_track is None:
                    matching_track = {
                        'box': box,
                        'first_frame': frame_number,
                        'last_frame': frame_number,
                        'frame_count': 0,
                        'captured': False,
                        'crop_candidates': [],
                    }
                    tracks.append(matching_track)
                matching_track['box'] = box
                matching_track['frame_count'] += 1
                crop = crop_plate(frame, box)
                remember_crop(matching_track, crop, crop_quality(crop, confidence))
                plate_width = crop.shape[1]
                previous_width = matching_track.get('last_plate_width', 0)
                peak_width = matching_track.get('peak_plate_width', 0)
                if plate_width > peak_width:
                    matching_track['peak_plate_width'] = plate_width
                    matching_track['peak_frame'] = frame_number
                    matching_track['shrinking_frames'] = 0
                elif plate_width < peak_width * 0.96:
                    matching_track['shrinking_frames'] = matching_track.get('shrinking_frames', 0) + 1
                elif plate_width >= previous_width:
                    matching_track['shrinking_frames'] = 0
                if matching_track.get('peak_plate_width', 0) >= MIN_CAPTURE_PLATE_WIDTH:
                    matching_track['near_enough'] = True
                matching_track['last_plate_width'] = plate_width
                matching_track['last_frame'] = frame_number

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'PLATE {confidence:.2f}', (box[0], max(24, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                ready_to_capture = (
                    matching_track['frame_count'] >= MIN_TRACK_FRAMES and
                    matching_track.get('near_enough', False) and
                    matching_track.get('shrinking_frames', 0) >= SHRINKING_FRAMES_TO_CAPTURE
                )
                if not matching_track['captured'] and ready_to_capture:
                    capture_number = queue_track_capture(
                        matching_track, capture_number, reader, ocr_executor, captures, pending_ocr
                    )

            stale_tracks = [track for track in tracks if frame_number - track['last_frame'] > frame_gap]
            for track in stale_tracks:
                if track.get('frame_count', 0) >= MIN_TRACK_FRAMES:
                    capture_number = queue_track_capture(
                        track, capture_number, reader, ocr_executor, captures, pending_ocr
                    )
            tracks = [track for track in tracks if frame_number - track['last_frame'] <= frame_gap]
            display = cv2.hconcat([make_panel(frame, captures), frame])
            cv2.imshow('Video ALPR - plates', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Flush tracks still visible at the end of a short video.
        for track in tracks:
            if track.get('frame_count', 0) >= MIN_TRACK_FRAMES:
                capture_number = queue_track_capture(
                    track, capture_number, reader, ocr_executor, captures, pending_ocr
                )
        for job in pending_ocr:
            try:
                text, confidence = job['future'].result()
            except Exception as error:
                text, confidence = '', 0.0
                print(f"OCR ล้มเหลว {job['filename']}: {error}")
            label = text or 'อ่านไม่ออก'
            job['entry'][1] = f"{job['number']}: {label}"
            results_file.write(
                f"ภาพที่ {job['number']} (ไฟล์ {job['filename']}): "
                f"{label} confidence={confidence:.3f}\n"
            )
            print(f"OCR {job['filename']} -> {label} confidence={confidence:.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ตรวจจับและอ่านป้ายทะเบียนจากไฟล์วิดีโอ')
    parser.add_argument('video', nargs='?', default=os.getenv('VIDEO_PATH'), help='พาธไฟล์วิดีโอ')
    parser.add_argument(
        '--imgsz',
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f'ขนาดภาพที่ส่งเข้า YOLO (ค่าเริ่มต้น {DEFAULT_IMAGE_SIZE}; ใช้ 1280 เพื่อความแม่นยำสูงสุด)',
    )
    parser.add_argument(
        '--detect-every',
        type=int,
        default=DEFAULT_DETECTION_INTERVAL,
        help=f'ตรวจจับ YOLO ทุกกี่เฟรม (ค่าเริ่มต้น {DEFAULT_DETECTION_INTERVAL}; ใช้ 1 เพื่อความแม่นยำสูงสุด)',
    )
    args = parser.parse_args()
    if not args.video:
        raise SystemExit('กรุณาระบุไฟล์วิดีโอ เช่น python webcam_detect.py video.mp4')
    process_video(args.video, image_size=args.imgsz, detection_interval=args.detect_every)
