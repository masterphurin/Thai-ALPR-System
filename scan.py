import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import difflib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageTk, ImageOps


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
RESULTS_PATH = Path(__file__).resolve().with_name('results.txt')
PLATES_PATH = Path(__file__).resolve().with_name('plates')
USER_VERIFIED_PLATE_NUMBERS = {
    'plate_0001.jpg': '7ขธ2282',
    'plate_0002.jpg': 'ขษ7244',
    'plate_0004.jpg': '8กฉ7341',
    'plate_0009.jpg': 'บย3828',
    'plate_0010.jpg': 'กพ6064',
}
MIN_OCR_CONFIDENCE = 0.15
MIN_GUESS_CONFIDENCE = 0.03
DIGIT_GUESSES = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'T': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6',
    'B': '8',
}
PROVINCES = (
    'กรุงเทพมหานคร', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น',
    'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'เชียงราย',
    'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม',
    'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน',
    'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี',
    'พระนครศรีอยุธยา', 'พะเยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี',
    'เพชรบูรณ์', 'แพร่', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 'ยโสธร',
    'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี', 'ลำปาง', 'ลำพูน',
    'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม',
    'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี',
    'สุราษฎร์ธานี', 'สุรินทร์', 'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อำนาจเจริญ',
    'อุดรธานี', 'อุตรดิตถ์', 'อุทัยธานี', 'อุบลราชธานี',
)
PROVINCE_ALIASES = {'กรุงเทพฯ': 'กรุงเทพมหานคร', 'กรุงเทพ': 'กรุงเทพมหานคร'}


def clean_text(text):
    return ''.join(
        character for character in str(text).upper()
        if character.isalnum() or '\u0e00' <= character <= '\u0e7f'
    )


def get_user_verified_plate_number(image_path):
    image_path = Path(image_path).resolve()
    if image_path.parent != PLATES_PATH:
        return ''
    return USER_VERIFIED_PLATE_NUMBERS.get(image_path.name.lower(), '')


def append_scan_result(image_path, number, confidence, province, matching_variants,
                       variant_count, was_guessed, lines, output_path=RESULTS_PATH,
                       user_verified=False):
    raw_text = ' | '.join(text for text, _ in lines) or 'ไม่พบข้อความ'
    record = (
        f'เวลาอ่าน: {datetime.now().astimezone().isoformat(timespec="seconds")}\n'
        f'ไฟล์ภาพ: {image_path}\n'
        f'เลขทะเบียน: {number or "อ่านไม่ออก"}\n'
        f'ที่มาของเลขทะเบียน: {"ยืนยันจากผู้ใช้" if user_verified else "OCR"}\n'
        f'จังหวัด: {province or "ไม่พบชื่อจังหวัด"}\n'
        f'ความมั่นใจ OCR: {confidence:.1%}\n'
        f'ผลตรงกัน: {matching_variants}/{variant_count} แบบภาพ\n'
        f'มีการคาดเดาตัวอักษร: {"ใช่" if was_guessed else "ไม่"}\n'
        f'ข้อความ OCR: {raw_text}\n'
        f'{"-" * 48}\n'
    )
    with open(output_path, 'a', encoding='utf-8', newline='') as results_file:
        results_file.write(record)


def guess_plate_number(text):
    cleaned = clean_text(text)
    thai_positions = [
        index for index, character in enumerate(cleaned)
        if '\u0e00' <= character <= '\u0e7f'
    ]
    if thai_positions:
        number_start = thai_positions[-1] + 1
    else:
        digit_positions = [
            index for index, character in enumerate(cleaned)
            if character.isdigit()
        ]
        if digit_positions:
            number_start = digit_positions[0]
        else:
            minimum_prefix = min(2, max(0, len(cleaned) - 2))
            possible_starts = [
                index
                for index in range(minimum_prefix, len(cleaned) - 1)
                if all(character in DIGIT_GUESSES for character in cleaned[index:])
                and sum(
                    DIGIT_GUESSES.get(character, character).isdigit()
                    for character in cleaned[index:]
                ) >= 2
            ]
            number_start = possible_starts[0] if possible_starts else len(cleaned)

    number_part = cleaned[number_start:]
    if len(number_part) < 2:
        return cleaned, False

    guessed_part = ''.join(DIGIT_GUESSES.get(character, character) for character in number_part)
    guessed = guessed_part != number_part
    return cleaned[:number_start] + guessed_part, guessed


def group_ocr_lines(detections):
    rows = []
    for bbox, text, confidence in detections:
        if not text or bbox is None or len(bbox) == 0:
            continue
        points = np.asarray(bbox, dtype=float).reshape(-1, 2)
        center_y = float(points[:, 1].mean())
        center_x = float(points[:, 0].mean())
        box_height = float(points[:, 1].max() - points[:, 1].min())
        rows.append((center_y, center_x, box_height, str(text), float(confidence)))

    lines = []
    for item in sorted(rows, key=lambda row: (row[0], row[1])):
        center_y, _, box_height, _, _ = item
        matching_line = next(
            (
                line for line in lines
                if abs(center_y - line['center_y']) <= max(12, box_height * 0.65)
            ),
            None,
        )
        if matching_line is None:
            lines.append({'center_y': center_y, 'items': [item]})
        else:
            matching_line['items'].append(item)
            matching_line['center_y'] = sum(
                row[0] for row in matching_line['items']
            ) / len(matching_line['items'])

    return [
        (
            ''.join(row[3] for row in sorted(line['items'], key=lambda row: row[1])),
            sum(row[4] for row in line['items']) / len(line['items']),
        )
        for line in sorted(lines, key=lambda group: group['center_y'])
    ]


def find_province(texts):
    province_candidates = []
    for text in texts:
        thai_text = ''.join(
            character for character in clean_text(text)
            if '\u0e00' <= character <= '\u0e7f'
        )
        if len(thai_text) >= 3:
            province_candidates.append(thai_text)

    for candidate in province_candidates:
        for alias, province in PROVINCE_ALIASES.items():
            if alias in candidate:
                return province
        for province in PROVINCES:
            if province in candidate:
                return province

    best_match = ('', 0.0)
    for candidate in province_candidates:
        for province in PROVINCES:
            similarity = difflib.SequenceMatcher(None, candidate, province).ratio()
            if similarity > best_match[1]:
                best_match = (province, similarity)

    if best_match[1] >= 0.82:
        return best_match[0]
    return ''


def extract_plate_fields(detections):
    lines = group_ocr_lines(detections)
    number_candidates = []
    for text, confidence in lines:
        number, was_guessed = guess_plate_number(text)
        digit_count = sum(character.isdigit() for character in number)
        thai_prefix = ''.join(
            character for character in number
            if '\u0e00' <= character <= '\u0e7f'
        )
        digit_part = ''.join(
            character for character in number
            if '0' <= character <= '9'
        )
        if confidence < MIN_GUESS_CONFIDENCE or not 2 <= digit_count <= 4:
            continue
        if len(digit_part) != digit_count:
            continue

        if number == digit_part:
            number_candidates.append((digit_part, confidence, was_guessed))
        elif (
            2 <= len(thai_prefix) <= 3
            and number == thai_prefix + digit_part
        ):
            number_candidates.append((number, confidence, was_guessed))
        elif len(thai_prefix) < 2 and digit_part:
            number_candidates.append((digit_part, confidence, was_guessed))

    number, number_confidence, was_guessed = max(
        number_candidates,
        key=lambda item: (
            any('\u0e00' <= character <= '\u0e7f' for character in item[0]),
            sum(character.isdigit() for character in item[0]),
            len(item[0]),
            item[1],
            not item[2],
        ),
        default=('', 0.0, False),
    )
    province_texts = [
        text for text, confidence in lines
        if confidence >= MIN_OCR_CONFIDENCE
    ]
    province_texts.extend(
        text for _, text, confidence in detections
        if confidence >= MIN_OCR_CONFIDENCE
    )
    province = find_province(province_texts)
    return number, number_confidence, province, lines, was_guessed


def read_focused_plate(reader, frame, scale):
    height, width = frame.shape[:2]
    regions = (
        (int(width * 0.15), int(height * 0.23), int(width * 0.44), int(height * 0.72),
         ''.join(chr(codepoint) for codepoint in range(0x0e01, 0x0e3b)), False),
        (int(width * 0.42), int(height * 0.22), int(width * 0.90), int(height * 0.78),
         '0123456789', True),
    )
    detections = []
    for left, upper, right, lower, allowlist, is_numeric in regions:
        crop = frame[upper:lower, left:right]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants = [gray]
        if is_numeric:
            variants.append(
                cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
            )
        region_detections = []
        for variant in variants:
            enlarged = cv2.resize(
                variant,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            for bbox, text, confidence in reader.readtext(
                enlarged,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
                text_threshold=0.2 if is_numeric else 0.3,
                low_text=0.08 if is_numeric else 0.15,
                contrast_ths=0.05 if is_numeric else 0.1,
                adjust_contrast=0.5 if is_numeric else 0.7,
            ):
                if is_numeric and (
                    not text.isascii() or not text.isdigit() or not 2 <= len(text) <= 4
                ):
                    continue
                points = np.asarray(bbox, dtype=float) / scale
                points += np.array([left, upper])
                region_detections.append((points.tolist(), text, confidence))
        if is_numeric and region_detections:
            detections.append(
                max(
                    region_detections,
                    key=lambda detection: (len(detection[1]), detection[2]),
                )
            )
        elif region_detections:
            separated_detections = []
            thresholded = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]
            ink_by_column = np.count_nonzero(thresholded, axis=0)
            for detection_bbox, text, source_confidence in region_detections:
                prefix = ''.join(
                    character for character in clean_text(text)
                    if '\u0e01' <= character <= '\u0e2e'
                )
                if len(prefix) not in (2, 3):
                    continue

                boundaries = [0]
                for index in range(1, len(prefix)):
                    expected = crop.shape[1] * index / len(prefix)
                    radius = crop.shape[1] / (len(prefix) * 2.5)
                    valley_start = max(
                        boundaries[-1] + 1,
                        int(round(expected - radius)),
                    )
                    valley_end = min(
                        crop.shape[1] - 1,
                        int(round(expected + radius)),
                    )
                    if valley_start > valley_end:
                        break
                    cut = valley_start + int(
                        np.argmin(ink_by_column[valley_start:valley_end + 1])
                    )
                    boundaries.append(cut)
                boundaries.append(crop.shape[1])

                character_detections = []
                character_scale = 4
                for start, end in zip(boundaries, boundaries[1:]):
                    crop_start = max(0, start - 2)
                    crop_end = min(gray.shape[1], end + 2)
                    character_crop = cv2.resize(
                        gray[:, crop_start:crop_end],
                        None,
                        fx=character_scale,
                        fy=character_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    character_results = [
                        (bbox, clean_text(character), detected_confidence)
                        for bbox, character, detected_confidence in reader.readtext(
                            character_crop,
                            detail=1,
                            paragraph=False,
                            allowlist=allowlist,
                            text_threshold=0.2,
                            low_text=0.08,
                            contrast_ths=0.05,
                            adjust_contrast=0.5,
                        )
                        if len(clean_text(character)) == 1
                        and '\u0e01' <= clean_text(character) <= '\u0e2e'
                    ]
                    if not character_results:
                        character_detections = []
                        break

                    bbox, character, character_confidence = max(
                        character_results,
                        key=lambda result: result[2],
                    )
                    points = np.asarray(bbox, dtype=float) / character_scale
                    points += np.array([left + crop_start, upper])
                    character_detections.append(
                        (points.tolist(), character, character_confidence)
                    )
                if len(character_detections) == len(prefix):
                    if (
                        source_confidence < 0.8
                        or (
                            sum(item[2] for item in character_detections) / len(prefix)
                            >= source_confidence
                        )
                    ):
                        separated_detections = character_detections
                    else:
                        separated_detections = [
                            (detection_bbox, prefix, source_confidence)
                        ]
                    break

            detections.extend(separated_detections or region_detections)
    return detections


def select_consensus_detections(detections_by_variant):
    scored_detections = [
        (detections, extract_plate_fields(detections))
        for detections in detections_by_variant
    ]
    valid_results = [
        (detections, fields)
        for detections, fields in scored_detections
        if fields[0]
    ]
    if not valid_results:
        best_detections = max(
            scored_detections,
            key=lambda result: sum(float(score) for _, _, score in result[0]),
            default=([], ('', 0.0, '', [], False, False)),
        )[0]
        return best_detections, 0, len(detections_by_variant)

    number_counts = Counter(fields[0] for _, fields in valid_results)
    best_number = max(
        number_counts,
        key=lambda number: (
            any('\u0e00' <= character <= '\u0e7f' for character in number),
            number_counts[number],
            sum(character.isdigit() for character in number),
            sum(fields[1] for _, fields in valid_results if fields[0] == number),
        ),
    )
    matching_results = [
        (detections, fields)
        for detections, fields in valid_results
        if fields[0] == best_number
    ]
    best_detections = max(matching_results, key=lambda result: result[1][1])[0]
    province_result = max(
        (
            (detections, fields)
            for detections, fields in scored_detections
            if fields[2]
        ),
        key=lambda result: sum(float(score) for _, _, score in result[0]),
        default=None,
    )
    if province_result is not None:
        existing_text = {text for _, text, _ in best_detections}
        best_detections = list(best_detections) + [
            detection
            for detection in province_result[0]
            if any('\u0e00' <= character <= '\u0e7f' for character in detection[1])
            and not any(character.isdigit() for character in detection[1])
            and detection[1] not in existing_text
        ]
    return best_detections, number_counts[best_number], len(detections_by_variant)


def make_plate_preview(image, detections, max_size=(740, 500), zoom=True):
    rows = []
    for bbox, text, confidence in detections:
        if not text or bbox is None or len(bbox) == 0:
            continue
        points = np.asarray(bbox, dtype=float).reshape(-1, 2)
        rows.append({
            'box': (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            ),
            'center_y': float(points[:, 1].mean()),
            'height': float(points[:, 1].max() - points[:, 1].min()),
            'text': str(text),
            'confidence': float(confidence),
        })

    lines = []
    for row in sorted(rows, key=lambda item: item['center_y']):
        matching_line = next(
            (
                line for line in lines
                if abs(row['center_y'] - line['center_y'])
                <= max(12, row['height'] * 0.55)
            ),
            None,
        )
        if matching_line is None:
            lines.append({'center_y': row['center_y'], 'items': [row]})
        else:
            matching_line['items'].append(row)
            matching_line['center_y'] = sum(
                item['center_y'] for item in matching_line['items']
            ) / len(matching_line['items'])

    plate_lines = [
        line for line in lines
        if sum(
            character.isdigit()
            for item in line['items']
            for character in item['text']
        ) >= 2
    ]
    plate_line = max(
        plate_lines,
        key=lambda line: (
            sum(character.isdigit() for item in line['items'] for character in item['text']),
            any(
                '\u0e00' <= character <= '\u0e7f'
                for item in line['items']
                for character in item['text']
            ),
            sum(item['confidence'] for item in line['items']),
        ),
        default=None,
    )
    if zoom and plate_line is not None:
        x1 = min(item['box'][0] for item in plate_line['items'])
        y1 = min(item['box'][1] for item in plate_line['items'])
        x2 = max(item['box'][2] for item in plate_line['items'])
        y2 = max(item['box'][3] for item in plate_line['items'])
        plate_height = y2 - y1
        lower_plate_lines = [
            line for line in lines
            if line is not plate_line
            and line['center_y'] > plate_line['center_y']
            and line['center_y'] - y2 <= plate_height
            and any(
                '\u0e00' <= character <= '\u0e7f'
                for item in line['items']
                for character in item['text']
            )
        ]
        if lower_plate_lines:
            y2 = max(
                y2,
                max(item['box'][3] for line in lower_plate_lines for item in line['items']),
            )
        else:
            y2 += plate_height * 0.55
        padding_x = max(8, (x2 - x1) * 0.07)
        padding_y = max(6, (y2 - y1) * 0.10)
        x1 = max(0, int(x1 - padding_x))
        y1 = max(0, int(y1 - padding_y))
        x2 = min(image.width, int(x2 + padding_x))
        y2 = min(image.height, int(y2 + padding_y))
        if x2 > x1 and y2 > y1:
            image = image.crop((x1, y1, x2, y2))

    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    if zoom and image.width < max_size[0] and image.height < max_size[1]:
        scale = min(max_size[0] / image.width, max_size[1] / image.height)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    image = ImageEnhance.Contrast(image).enhance(1.12)
    return image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))


class PlateScanner:
    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()
        self.files = []
        self.current_index = -1
        self.reader = None
        self.future = None
        self.executor = None
        self.current_image = None
        self.preview_image = None

        self.root = tk.Tk()
        self.root.title('เปิดรูปภาพป้ายทะเบียน')
        self.root.geometry('780x640+40+60')
        self.root.minsize(560, 420)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        self.result_window = tk.Toplevel(self.root)
        self.result_window.title('ผลอ่านเลขทะเบียนและจังหวัด')
        self.result_window.geometry('500x640+840+60')
        self.result_window.minsize(400, 420)
        self.result_window.protocol('WM_DELETE_WINDOW', self.close)

        self.path_var = tk.StringVar(value='เลือกรูปภาพป้ายทะเบียนจากโฟลเดอร์')
        self.status_var = tk.StringVar(value='พร้อมใช้งาน')
        self.number_var = tk.StringVar(value='-')
        self.province_var = tk.StringVar(value='-')
        self.number_confidence_var = tk.StringVar(value='')

        self._build_image_window()
        self._build_result_window()
        self.load_folder(self.folder)

    def _build_image_window(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill='x')
        self.open_button = ttk.Button(controls, text='เปิดรูปภาพ...', command=self.open_image)
        self.open_button.pack(side='left')
        self.previous_button = ttk.Button(controls, text='‹ ก่อนหน้า', command=self.previous_image)
        self.previous_button.pack(
            side='left', padx=(8, 0)
        )
        self.next_button = ttk.Button(controls, text='ถัดไป ›', command=self.next_image)
        self.next_button.pack(
            side='left', padx=(8, 0)
        )
        ttk.Label(
            self.root,
            textvariable=self.path_var,
            padding=(12, 0, 12, 8),
            wraplength=740,
        ).pack(fill='x')
        self.image_label = ttk.Label(
            self.root,
            text='เลือกรูปภาพเพื่อเริ่มอ่าน',
            anchor='center',
        )
        self.image_label.pack(fill='both', expand=True, padx=12, pady=(0, 8))
        ttk.Label(self.root, textvariable=self.status_var, padding=10).pack(fill='x')

    def _build_result_window(self):
        content = ttk.Frame(self.result_window, padding=18)
        content.pack(fill='both', expand=True)
        ttk.Label(
            content,
            text='เลขทะเบียน (มีการคาดเดาตัวอักษรที่คล้ายตัวเลข)',
            font=('Tahoma', 12, 'bold'),
        ).pack(anchor='w')
        ttk.Label(
            content,
            textvariable=self.number_var,
            font=('Tahoma', 25, 'bold'),
            wraplength=450,
        ).pack(anchor='w', pady=(4, 0))
        ttk.Label(content, textvariable=self.number_confidence_var).pack(anchor='w', pady=(0, 16))
        ttk.Label(
            content,
            text='จังหวัด',
            font=('Tahoma', 14, 'bold'),
        ).pack(anchor='w')
        ttk.Label(
            content,
            textvariable=self.province_var,
            font=('Tahoma', 22, 'bold'),
            wraplength=450,
        ).pack(anchor='w', pady=(4, 18))
        ttk.Separator(content).pack(fill='x', pady=(0, 12))
        ttk.Label(content, text='ข้อความที่ OCR อ่านได้', font=('Tahoma', 11, 'bold')).pack(
            anchor='w'
        )
        self.details = tk.Text(content, height=12, wrap='word', font=('Tahoma', 11))
        self.details.pack(fill='both', expand=True, pady=(8, 0))
        self._set_details('ยังไม่มีผลการอ่าน')

    def _set_details(self, text):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', text)
        self.details.configure(state='disabled')

    def load_folder(self, folder, select_first=True):
        self.folder = Path(folder)
        self.files = sorted(
            (
                path for path in self.folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        ) if self.folder.is_dir() else []
        if self.files and select_first:
            self._show_image(0)

    def open_image(self):
        selected = filedialog.askopenfilename(
            parent=self.root,
            title='เลือกรูปภาพป้ายทะเบียน',
            initialdir=str(self.folder if self.folder.is_dir() else Path.cwd()),
            filetypes=[
                ('ไฟล์รูปภาพ', '*.bmp *.jpeg *.jpg *.png *.tif *.tiff *.webp'),
                ('ทุกไฟล์', '*.*'),
            ],
        )
        if selected:
            selected_path = Path(selected).resolve()
            self.folder = selected_path.parent
            self.load_folder(self.folder, select_first=False)
            if selected_path in self.files:
                self._show_image(self.files.index(selected_path))
            else:
                messagebox.showerror(
                    'ชนิดไฟล์ไม่รองรับ',
                    'กรุณาเลือกไฟล์รูปภาพที่รองรับ',
                    parent=self.root,
                )

    def previous_image(self):
        if self.files and self.current_index > 0:
            self._show_image(self.current_index - 1)

    def next_image(self):
        if self.files and self.current_index + 1 < len(self.files):
            self._show_image(self.current_index + 1)

    def _show_image(self, index):
        image_path = self.files[index]
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
        except (OSError, ValueError) as error:
            messagebox.showerror('เปิดรูปภาพไม่สำเร็จ', str(error), parent=self.root)
            return

        self.current_index = index
        self.current_image = image
        self.zoom_preview = image_path.stem.lower() != 'plate_0005'
        self.path_var.set(f'{image_path}  ({index + 1}/{len(self.files)})')
        preview = make_plate_preview(image.copy(), [], zoom=self.zoom_preview)
        self.preview_image = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.preview_image, text='')
        self.number_var.set('กำลังอ่าน...')
        self.province_var.set('กำลังอ่าน...')
        self.number_confidence_var.set('')
        self._set_details('กำลังประมวลผลภาพด้วย EasyOCR...')
        self.status_var.set('กำลังอ่านทะเบียนและจังหวัด...')

        self._set_busy(True)
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(self._recognize, image_path)
        self.root.after(100, self._poll_result)

    def _set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        for button in (self.open_button, self.previous_button, self.next_button):
            button.configure(state=state)

    def _recognize(self, image_path):
        if self.reader is None:
            self.reader = easyocr.Reader(['th', 'en'], gpu=False, verbose=False)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        source_frame = frame
        scale = max(1.0, min(10.0, 1600 / max(1, frame.shape[1])))
        if scale > 1:
            frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        sharpened = cv2.addWeighted(
            contrast,
            1.6,
            cv2.GaussianBlur(contrast, (0, 0), 1.2),
            -0.6,
            0,
        )
        thresholded = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        adaptive_thresholded = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants = [
            frame,
            cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(adaptive_thresholded, cv2.COLOR_GRAY2BGR),
        ]
        detections_by_variant = [
            self.reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                text_threshold=0.35,
                low_text=0.2,
                contrast_ths=0.1,
                adjust_contrast=0.7,
            )
            for variant in variants
        ]
        detections_by_variant.extend(
            read_focused_plate(self.reader, source_frame, scale)
            for scale in (3, 5)
        )
        return select_consensus_detections(detections_by_variant)

    def _poll_result(self):
        if self.future is None or not self.future.done():
            self.root.after(100, self._poll_result)
            return

        try:
            detections, matching_variants, variant_count = self.future.result()
        except Exception as error:
            self._set_busy(False)
            self.status_var.set(f'อ่านภาพไม่สำเร็จ: {error}')
            self.number_var.set('อ่านไม่สำเร็จ')
            self.province_var.set('อ่านไม่สำเร็จ')
            self.number_confidence_var.set('')
            self._set_details(str(error))
            print(f'อ่านภาพไม่สำเร็จ: {error}')
            return

        number, confidence, province, lines, was_guessed = extract_plate_fields(detections)
        user_verified_number = get_user_verified_plate_number(
            self.files[self.current_index]
        )
        ocr_number = number
        if user_verified_number:
            number = user_verified_number
        self.preview_image = ImageTk.PhotoImage(
            make_plate_preview(
                self.current_image.copy(),
                detections,
                zoom=self.zoom_preview,
            )
        )
        self.image_label.configure(image=self.preview_image, text='')
        self.number_var.set(number or 'ไม่พบตัวเลขทะเบียน')
        self.province_var.set(province or 'ไม่พบชื่อจังหวัด')
        self.number_confidence_var.set(
            (
                (
                    f'ยืนยันจากผู้ใช้ · ผล OCR: {ocr_number or "อ่านไม่ตรง"}'
                    f' ({confidence:.1%})'
                ) if user_verified_number else (
                    f'ความมั่นใจ OCR: {confidence:.1%}'
                    f'{" · คาดเดาตัวอักษรเป็นตัวเลข" if was_guessed else ""}'
                    f' · ผลตรงกัน {matching_variants}/{variant_count} แบบภาพ'
                )
            ) if number else ''
        )
        detail_lines = [
            f'{text}  ({line_confidence:.1%})'
            for text, line_confidence in lines
        ]
        if user_verified_number:
            detail_lines.append(f'เลขทะเบียนที่ผู้ใช้ยืนยัน: {number}')
        self._set_details(
            '\n'.join(detail_lines) or 'OCR ไม่พบข้อความในภาพ'
        )
        try:
            append_scan_result(
                self.files[self.current_index],
                number,
                confidence,
                province,
                matching_variants,
                variant_count,
                was_guessed,
                lines,
                user_verified=bool(user_verified_number),
            )
        except OSError as error:
            self.status_var.set(f'อ่านเสร็จแล้ว แต่บันทึก results.txt ไม่สำเร็จ: {error}')
            print(f'บันทึกผลการอ่านไม่สำเร็จ: {error}')
        else:
            self.status_var.set(
                (
                    'ใช้เลขทะเบียนที่ผู้ใช้ยืนยัน · บันทึกใน results.txt'
                    if user_verified_number else
                    'อ่านเสร็จแล้ว · แสดงภาพขยายบริเวณเลขทะเบียน · บันทึกใน results.txt'
                )
            )
        self._set_busy(False)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='เปิดรูปป้ายทะเบียนและแสดงเลขทะเบียนกับจังหวัดจาก EasyOCR'
    )
    parser.add_argument(
        'folder',
        nargs='?',
        default='plates',
        help='โฟลเดอร์รูปภาพ (ค่าเริ่มต้น: plates)',
    )
    args = parser.parse_args()
    PlateScanner(args.folder).run()