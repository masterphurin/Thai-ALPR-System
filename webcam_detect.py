import cv2
import os
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
print(f"\n✅ เชื่อมต่อกล้องหมายเลข {cam_id} สำเร็จ!")
print("--- ระบบพร้อมใช้งาน (รันครบข้อ 1-4) ---")
print(">> นำป้ายทะเบียนมาส่องที่กล้อง")
print(">> กด 'c' เพื่อ Capture, อ่านตัวอักษร และบันทึกลง Text ไฟล์")
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
    
    if key == ord('c'):
        img_count += 1
        
        # ข้อ 2: บันทึกภาพที่ Capture
        filename = f"{save_folder}/plate_angle_{img_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"\n📸 แชะ! บันทึกภาพแล้ว: {filename}")
        
        # ข้อ 3 & 4: อ่านตัวอักษรและเซฟไฟล์
        if len(detected_boxes) > 0:
            for (x1, y1, x2, y2) in detected_boxes:
                cropped_plate = frame[y1:y2, x1:x2]
                
                # ข้อ 3: อ่านด้วย Algorithm (EasyOCR)
                ocr_results = reader.readtext(cropped_plate)
                plate_text = ""
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.1:
                        plate_text += text + " "
                
                plate_text = plate_text.strip()
                if plate_text:
                    print(f"✅ อ่านป้ายได้: {plate_text}")
                    # ข้อ 4: เซฟลง Text ไฟล์
                    with open("results.txt", "a", encoding="utf-8") as f:
                        f.write(f"ภาพที่ {img_count} (ไฟล์ {filename}): {plate_text}\n")
                    print("📝 บันทึกผลลัพธ์ลงไฟล์ results.txt เรียบร้อย")
                else:
                    print("❌ ระบบเห็นป้าย แต่อ่านตัวหนังสือไม่ออก ลองขยับมุมใหม่ครับ")
        else:
            print("⚠️ จังหวะที่กดถ่าย ไม่มีกรอบสีเขียวจับป้ายอยู่")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()