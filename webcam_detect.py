import cv2
import os
from ultralytics import YOLO

save_folder = 'captured_plates'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print("กำลังโหลดโมเดล YOLO...")
plate_detector = YOLO('HurricaneOD_beta.pt')

cap = cv2.VideoCapture(0)
img_count = 0

print("\n--- ระบบพร้อมใช้งาน ---")
print(">> นำป้ายทะเบียน (หรือเปิดรูปป้ายทะเบียนในมือถือ) มาส่องที่กล้อง")
print(">> กด 'c' เพื่อ Capture ภาพ")
print(">> กด 'q' เพื่อปิดโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถดึงภาพจากกล้องได้")
        break

    results = plate_detector(frame, verbose=False)[0]

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Plate Detected", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Webcam - License Plate Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        img_count += 1
        filename = f"{save_folder}/plate_angle_{img_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 แชะ! บันทึกภาพแล้ว: {filename}")

    elif key == ord('q'):
        break

# คืนทรัพยากรระบบ
cap.release()
cv2.destroyAllWindows()