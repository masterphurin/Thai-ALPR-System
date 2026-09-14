import os
import cv2
import easyocr
from ultralytics import YOLO

# โหลดโมเดลทั้งสองตัว (YOLO หาป้าย + EasyOCR อ่านอักษร)
print("กำลังโหลดโมเดล AI...")
plate_detector = YOLO('HurricaneOD_beta.pt')
reader = easyocr.Reader(['th', 'en'], gpu=False)

folder_path = './images'
if not os.path.exists(folder_path):
    exit()

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            print(f"\n[{filename}]")
            
            # 1. สั่งให้ YOLO ค้นหาตำแหน่งป้ายทะเบียนก่อน
            results = plate_detector(img, verbose=False)[0]
            
            if len(results.boxes) > 0:
                for box in results.boxes:
                    # ดึงพิกัด (x, y) ของป้ายทะเบียน
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 2. ตัดภาพ (Crop) เอาเฉพาะกรอบป้ายทะเบียน
                    cropped_plate = img[y1:y2, x1:x2]
                    
                    # 3. ให้ EasyOCR อ่านเฉพาะภาพป้ายที่ตัดมา
                    ocr_results = reader.readtext(cropped_plate)
                    
                    print("--- ข้อความที่อ่านได้ ---")
                    for (bbox, text, prob) in ocr_results:
                        if prob > 0.1:
                            print(f"-> {text} ({prob*100:.1f}%)")
                    
                    # วาดกรอบสีเขียวบนรูปใหญ่
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # โชว์รูปป้ายที่ถูกตัดออกมาให้ดูด้วย
                    cv2.imshow('Cropped Plate', cropped_plate)
            else:
                print("❌ YOLO มองไม่เห็นป้ายทะเบียน")
            
            cv2.imshow('Main Image', img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

cv2.destroyAllWindows()