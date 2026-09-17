import cv2
import os
import re
import easyocr
from ultralytics import YOLO

save_folder = 'captured_plates'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("กำลังโหลดโมเดล YOLO และ EasyOCR...")
plate_detector = YOLO('HurricaneOD_beta.pt')
reader = easyocr.Reader(['th', 'en'], gpu=False)

# ใช้ CAM_ID ระบุหมายเลขกล้องได้ เช่น PowerShell: $env:CAM_ID=1
configured_cam_id = os.getenv('CAM_ID')
camera_ids = [int(configured_cam_id)] if configured_cam_id else [0, 1, 2, 3]
camera_ids = list(dict.fromkeys(camera_ids))

cap = None
cam_id = None
for candidate_id in camera_ids:
    # CAP_DSHOW ช่วยให้กล้อง USB และ Iriun ทำงานได้เสถียรกว่าบน Windows
    candidate = cv2.VideoCapture(candidate_id, cv2.CAP_DSHOW)
    if not candidate.isOpened():
        candidate.release()
        candidate = cv2.VideoCapture(candidate_id)

    if candidate.isOpened():
        cap = candidate
        cam_id = candidate_id
        break
    candidate.release()

if cap is None:
    print("\n❌ ไม่พบกล้องที่สามารถเชื่อมต่อได้")
    print(f"หมายเลขที่ลอง: {camera_ids}")
    print("ตรวจสอบว่าเสียบกล้องแล้ว ปิดโปรแกรมอื่นที่กำลังใช้กล้อง และเปิด Iriun ก่อน")
    print("หากต้องการระบุหมายเลขเอง ให้ตั้งค่า CAM_ID เช่น: $env:CAM_ID=1")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

img_count = 0
plate_was_detected = False
read_confirmation_count = 0
required_read_confirmations = 2
last_recognized_texts = []
print(f"\n✅ เชื่อมต่อกล้องหมายเลข {cam_id} สำเร็จ!")
print("--- ระบบพร้อมใช้งาน (รันครบข้อ 1-4) ---")
print(">> นำป้ายทะเบียนมาส่องที่กล้อง")
print(">> ระบบจะ Capture, อ่านตัวอักษร และบันทึกให้อัตโนมัติเมื่อพบป้าย")
print(">> กด 'q' เพื่อปิดโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        print("สัญญาณภาพหลุด ไม่สามารถดึงภาพจากกล้องได้")
        break

    # ข้อ 1: Detect ป้ายทะเบียน
    results = plate_detector(frame, verbose=False)[0]
    detected_boxes = []

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_boxes.append((x1, y1, x2, y2))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Plate Detected", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Webcam - License Plate Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Capture once per detection event, then wait until the plate disappears
    # before allowing another automatic capture.
    if detected_boxes and not plate_was_detected:
        recognized_texts = []

        # อ่านก่อนบันทึกภาพ เพื่อรอเฟรมที่เห็นตัวอักษรชัดเจน
        for (x1, y1, x2, y2) in detected_boxes:
            cropped_plate = frame[y1:y2, x1:x2]
            ocr_results = reader.readtext(cropped_plate)
            plate_text = ""
            text_confidences = []
            for (bbox, text, prob) in ocr_results:
                cleaned_text = re.sub(r"[^ก-๙A-Za-z0-9]", "", text)
                if prob > 0.35 and len(cleaned_text) >= 3:
                    plate_text += text + " "
                    text_confidences.append(prob)

            plate_text = plate_text.strip()
            if plate_text and sum(text_confidences) / len(text_confidences) >= 0.45:
                recognized_texts.append(plate_text)

        if recognized_texts:
            if recognized_texts == last_recognized_texts:
                read_confirmation_count += 1
            else:
                read_confirmation_count = 1
                last_recognized_texts = recognized_texts

            if read_confirmation_count >= required_read_confirmations:
                img_count += 1
                plate_was_detected = True
                filename = f"{save_folder}/plate_angle_{img_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n📸 ยืนยันตัวอักษรแล้วและบันทึกภาพ: {filename}")

                with open("results.txt", "a", encoding="utf-8") as f:
                    for plate_text in recognized_texts:
                        print(f"✅ อ่านป้ายได้: {plate_text}")
                        f.write(f"ภาพที่ {img_count} (ไฟล์ {filename}): {plate_text}\n")
                print("📝 บันทึกผลลัพธ์ลงไฟล์ results.txt เรียบร้อย")
                read_confirmation_count = 0
                last_recognized_texts = []
            else:
                print(f"⏳ อ่านได้แล้ว กำลังยืนยันอีก {required_read_confirmations - read_confirmation_count} เฟรม...")
        else:
            read_confirmation_count = 0
            last_recognized_texts = []
            print("⏳ พบป้ายแล้ว แต่ยังอ่านไม่ชัด กำลังรอเฟรมถัดไป...")
    elif not detected_boxes:
        plate_was_detected = False
        read_confirmation_count = 0
        last_recognized_texts = []

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()