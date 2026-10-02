import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


PLATE_DIR = 'plates'
MODEL_PATH = 'license_plate_detector.pt'
MIN_PLATE_CONFIDENCE = 0.50
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_DETECTION_INTERVAL = 1
MIN_TRACK_FRAMES = 6
MAX_CROP_CANDIDATES = 5
PLATE_PADDING_RATIO = 0.10
MIN_CAPTURE_PLATE_WIDTH = 90
SHRINKING_FRAMES_TO_CAPTURE = 2
PLAYBACK_SPEED = 0.90
CAPTURE_UPSCALE_FACTOR = 3


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


def enhance_capture_crop(crop):
    """Enlarge the saved plate crop and apply a mild unsharp mask."""
    if crop.size == 0:
        return crop

    enlarged = cv2.resize(
        crop,
        None,
        fx=CAPTURE_UPSCALE_FACTOR,
        fy=CAPTURE_UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )
    blurred = cv2.GaussianBlur(enlarged, (0, 0), 1.0)
    return cv2.addWeighted(enlarged, 1.35, blurred, -0.35, 0)


def queue_track_capture(track, capture_number, captures):
    """Save the best plate crop without attempting to read its text."""
    if track.get('captured') or track.get('best_crop') is None:
        return capture_number

    capture_number += 1
    crop = enhance_capture_crop(track['best_crop'])
    filename = os.path.join(PLATE_DIR, f'plate_{capture_number:04d}.jpg')
    if not cv2.imwrite(filename, crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        print(f'บันทึกรูปไม่สำเร็จ: {filename}')
        return capture_number - 1

    captures.append(crop.copy())
    track['captured'] = True
    print(f'แคปภาพแล้ว: {filename}')
    return capture_number


def make_panel(frame, captures, panel_width=300):
    panel = cv2.resize(frame, (panel_width, frame.shape[0]))
    cv2.rectangle(panel, (0, 0), (panel_width, 42), (30, 30, 30), -1)
    cv2.putText(panel, 'CAPTURES', (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    y = 55
    for image in reversed(captures):
        if y + 105 > panel.shape[0]:
            break
        thumbnail = cv2.resize(image, (panel_width - 20, 78))
        panel[y:y + 78, 10:panel_width - 10] = thumbnail
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

    print('กำลังโหลดโมเดล YOLO...')
    plate_detector = YOLO(MODEL_PATH)
    captures = deque(maxlen=8)
    tracks = []
    frame_number = 0
    capture_number = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(video_fps) or video_fps <= 0:
        video_fps = 30.0
    frame_gap = max(1, int(video_fps * 0.7))
    target_frame_duration = 1.0 / (video_fps * PLAYBACK_SPEED)

    while True:
        frame_started = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
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
            matching_track['confidence'] = confidence
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

            ready_to_capture = (
                matching_track['frame_count'] >= MIN_TRACK_FRAMES and
                matching_track.get('near_enough', False) and
                matching_track.get('shrinking_frames', 0) >= SHRINKING_FRAMES_TO_CAPTURE
            )
            if not matching_track['captured'] and ready_to_capture:
                capture_number = queue_track_capture(matching_track, capture_number, captures)

        for track in tracks:
            if frame_number - track['last_frame'] >= max(1, detection_interval):
                continue
            box = track['box']
            confidence = track.get('confidence', 0.0)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'PLATE {confidence:.2f}', (box[0], max(24, box[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        stale_tracks = [track for track in tracks if frame_number - track['last_frame'] > frame_gap]
        for track in stale_tracks:
            if track.get('frame_count', 0) >= MIN_TRACK_FRAMES:
                capture_number = queue_track_capture(track, capture_number, captures)
        tracks = [track for track in tracks if frame_number - track['last_frame'] <= frame_gap]
        display = cv2.hconcat([make_panel(frame, captures), frame])
        cv2.imshow('Video - image capture', display)
        elapsed = time.perf_counter() - frame_started
        wait_ms = max(1, int((target_frame_duration - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    for track in tracks:
        if track.get('frame_count', 0) >= MIN_TRACK_FRAMES:
            capture_number = queue_track_capture(track, capture_number, captures)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ตรวจจับและแคปภาพป้ายทะเบียนจากไฟล์วิดีโอ')
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
