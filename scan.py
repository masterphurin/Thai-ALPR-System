import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageTk, ImageOps


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}
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
        if confidence >= MIN_GUESS_CONFIDENCE and digit_count >= 2:
            number_candidates.append((number, confidence, was_guessed))

    number, number_confidence, was_guessed = max(
        number_candidates,
        key=lambda item: (
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


class PlateScanner:
    def __init__(self, folder):
        self.folder = Path(folder).expanduser().resolve()
        self.files = []
        self.current_index = -1
        self.reader = None
        self.future = None
        self.executor = None
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
        ttk.Label(content, text='จังหวัด', font=('Tahoma', 14, 'bold')).pack(anchor='w')
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
        self.path_var.set(f'{image_path}  ({index + 1}/{len(self.files)})')
        preview_scale = min(2.5, 740 / image.width, 500 / image.height)
        preview = image.resize(
            (
                max(1, int(image.width * preview_scale)),
                max(1, int(image.height * preview_scale)),
            ),
            Image.Resampling.LANCZOS,
        )
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
        scale = max(1, min(10, 1000 // max(1, frame.shape[1])))
        variants = [frame]
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
            variants = [
                frame,
                cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR),
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
        scored_detections = [
            (detections, extract_plate_fields(detections))
            for detections in detections_by_variant
        ]
        return max(
            scored_detections,
            key=lambda result: (
                sum(character.isdigit() for character in result[1][0]),
                result[1][1],
                sum(float(score) for _, _, score in result[0]),
            ),
            default=([], ('', 0.0, '', [], False)),
        )[0]

    def _poll_result(self):
        if self.future is None or not self.future.done():
            self.root.after(100, self._poll_result)
            return

        try:
            detections = self.future.result()
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
        self.number_var.set(number or 'ไม่พบตัวเลขทะเบียน')
        self.province_var.set(province or 'ไม่พบชื่อจังหวัด')
        self.number_confidence_var.set(
            (
                f'ความมั่นใจ OCR: {confidence:.1%}'
                f'{" · คาดเดาตัวอักษรเป็นตัวเลข" if was_guessed else ""}'
            ) if number else ''
        )
        self._set_details(
            '\n'.join(
                f'{text}  ({line_confidence:.1%})'
                for text, line_confidence in lines
            ) or 'OCR ไม่พบข้อความในภาพ'
        )
        self.status_var.set('อ่านเสร็จแล้ว')
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