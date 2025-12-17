from ultralytics import YOLO
import easyocr
import cv2
import os
import requests 
import re

# ===============================
# ตั้งค่า API
# ===============================
# ใส่ URL ที่ได้จาก Google Apps Script (Deploy > Web App)
API_URL = "https://script.google.com/macros/s/AKfycbyGHuocj5zSach0XTWra9P0D9yy9KvFxJYLuS1HZpZ5NgjcohRPwhYMR1HSb-DbavHU/exec" 
API_KEY = "ASHI_CREAM" 

# ===============================
# ฟังก์ชันส่งข้อมูลเข้า Dashboard
# ===============================
# --- แก้ไขช่วงบรรทัด 18-30 ---

# 1. เพิ่มตัวแปร province ในวงเล็บ
def send_to_dashboard(plate_number, province):
    """
    ส่งเลขทะเบียนและจังหวัดไปตรวจสอบสิทธิ์และบันทึก Log ลง Google Sheet
    """
    print(f"🚀 กำลังส่งข้อมูล: {plate_number} | {province} ไปยัง Dashboard...")

    try:
        # ใช้ action='check' เพื่อบันทึก Log 
        # 2. เพิ่ม key "province" ลงไปใน params
        response = requests.get(API_URL, params={
            "action": "check",
            "license": plate_number,
            "province": province,  # <--- เพิ่มบรรทัดนี้สำคัญมาก!
            "key": API_KEY
        })
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('access', 'Unknown')
            owner = data.get('data', {}).get('owner', '-')
            
            if result == "Allow":
                print(f"✅ ผลลัพธ์: อนุญาต (Allow) | เจ้าของ: {owner}")
            else:
                print(f"⛔ ผลลัพธ์: ปฏิเสธ (Deny) | ติดต่อ รปภ.")
            return True
        else:
            print(f"❌ Server Error: {response.status_code}")
            return False

    except Exception as e:
        print(f"⚠️ เชื่อมต่อเน็ตไม่ได้: {e}")
        return False

# ===============================
# เริ่มต้นระบบ AI
# ===============================
model = YOLO("license_plate.pt")
reader = easyocr.Reader(['th', 'en'], gpu=False)

image_folder = "images"
valid_ext = (".jpg", ".jpeg", ".png")

print("--- เริ่มต้นระบบตรวจจับ ---")

for filename in os.listdir(image_folder):
    if not filename.lower().endswith(valid_ext):
        continue

    image_path = os.path.join(image_folder, filename)
    print(f"\n📷 กำลังประมวลผลรูป: {filename}")

    img = cv2.imread(image_path)
    if img is None: continue

    # 1) YOLO Detect
    results = model(img)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] != "license_plate":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            plate_img = img[y1:y2, x1:x2]
            if plate_img.size == 0: continue

            # 2) OCR Read
            texts = reader.readtext(plate_img)

            # เปลี่ยนจาก "" เป็น " " (เคาะวรรค) เพื่อไม่ให้ตัวอักษรติดกันเกินไป
            raw_text = " ".join([t[1] for t in texts]).strip()

            # กำหนดค่าเริ่มต้น
            plate_num = raw_text
            province = "ไม่ระบุ"

            # ใช้ Regex แยก "เลขทะเบียน" กับ "จังหวัด"
            # เงื่อนไข: หาชุดข้อความที่จบด้วยตัวเลข (เลขทะเบียน) แล้วแยกส่วนหลังออก (จังหวัด)
            match = re.match(r"(.*?[\d]+)\s*(.*)", raw_text)
            if match:
                plate_num = match.group(1).strip() # ได้เลขทะเบียน
                province = match.group(2).strip()  # ได้จังหวัด

            # ถ้าจังหวัดว่างเปล่า (AI อ่านไม่เจอ) ให้ใส่ขีด - หรือปล่อยว่าง
            if province == "":
                province = "ไม่ระบุ"

            # ถ้าอ่านเจอตัวอักษร ให้ส่งข้อมูลทันที
            if len(plate_num) > 2:
                print(f"🚗 อ่านป้ายได้: {plate_num} | จังหวัด: {province}")

                # *** ส่งข้อมูลแบบแยก 2 ค่า (ต้องไปแก้ฟังก์ชันนี้ด้วย ดูข้อ 3) ***
                send_to_dashboard(plate_num, province) 

            # วาดกรอบ
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, raw_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # บันทึกรูปผลลัพธ์ (ถ้าต้องการดู)
    # cv2.imwrite(f"output_{filename}", img)