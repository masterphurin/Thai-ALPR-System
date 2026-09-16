import cv2
import os
import argparse
from ultralytics import YOLO

save_folder = 'captured_plates'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def open_camera(camera_index):
    """Open a Windows camera using DirectShow, including Iriun Virtual Camera."""
    if camera_index == 'auto':
        available_cameras = []
        for index in range(6):
            test_camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if test_camera.isOpened():
                available_cameras.append(index)
                test_camera.release()

        if not available_cameras:
            return None, []

        # Iriun is commonly the second camera exposed by Windows.
        camera_index = 1 if 1 in available_cameras else available_cameras[0]

    camera = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    return camera if camera.isOpened() else None, [camera_index]


parser = argparse.ArgumentParser(description='ตรวจจับป้ายทะเบียนจากกล้อง Iriun')
parser.add_argument(
    '--camera',
    default=os.environ.get('IRIUN_CAMERA_INDEX', 'auto'),
    help='หมายเลขกล้อง เช่น 1 หรือ auto (ค่าเริ่มต้น: auto)',
)
args = parser.parse_args()

print("กำลังโหลดโมเดล YOLO...")
plate_detector = YOLO('HurricaneOD_beta.pt')

cap, detected_cameras = open_camera(args.camera)
if cap is None:
    print("ไม่สามารถเปิดกล้องได้")
    print("ตรวจสอบว่าเปิด Iriun Webcam บนมือถือและ Iriun Webcam บนคอมพิวเตอร์แล้ว")
    print("ลองระบุหมายเลขกล้องด้วยคำสั่ง: python webcam_detect.py --camera 1")
    raise SystemExit(1)

print(f"เชื่อมต่อกล้องหมายเลข {detected_cameras[0]} แล้ว")
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