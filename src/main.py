import torch
import time
import pytz
import requests
import numpy as np
from datetime import datetime, timedelta
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import matplotlib.pyplot as plt


LINE_ACCESS_TOKEN = "ํLINE_ACCESS_TOKEN"
LINE_API_URL = "https://api.line.me/v2/bot/message/push"
USER_ID = "๊USER_ID"
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
PARENT_FOLDER_ID = "PARENT_FOLDER_ID"

thai_tz = pytz.timezone("Asia/Bangkok")

sam_checkpoint = "models/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

ngrok_url = "https://XXXX.ngrok-free.app"
full_url = ngrok_url + "/snapshot"

input_point = np.array([[496, 426]])  # ระบุพิกัดของจุดที่สนใจ
input_label = np.array([1])


def upload_to_drive(file_path, file_name, folder_id):
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="image/jpeg")
    file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    file_id = file.get("id")
    drive_service.permissions().create(fileId=file_id, body={"role": "reader", "type": "anyone"}).execute()

    return f"https://drive.google.com/uc?id={file_id}"

def capture_and_process():
    now = datetime.now(thai_tz).strftime('%H:%M:%S')
    print(f"[{now}] เริ่มกระบวนการประมวลผล")

    response = requests.get(full_url)
    if response.status_code == 200:
        with open("snapshot.jpg", "wb") as f:
            f.write(response.content)
        print("บันทึก snapshot.jpg สำเร็จ!")
    else:
        print("ดึงรูปภาพล้มเหลว, status code:", response.status_code)
        return None, None, None  # ไม่ได้รูป → คืน None

    image = np.array(Image.open("snapshot.jpg"))
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    mask = masks[0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    object_height = bottom - top
    Output = 11 - ((object_height) / 35.89)
    water_level = float("{:.2f}".format(Output))

    print(f"ระดับน้ำที่คำนวณได้: {water_level} เมตร")

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.contour(mask, colors="red", linewidths=2)
    plt.gca().add_patch(plt.Rectangle(
    (np.min(np.where(cols)), top),
    np.max(np.where(cols)) - np.min(np.where(cols)),
    object_height,
    edgecolor="blue", facecolor="none", linewidth=2
    ))
    plt.title("Detected Object with Height")
    plt.axis("off")
    plt.show()

    if water_level < 4.6:
        waterphase = 1
    elif 4.6 <= water_level < 5.0:
        waterphase = 2
    elif 5.0 <= water_level < 5.5:
        waterphase = 3
    elif 5.5 <= water_level < 6.0:
        waterphase = 4
    elif 6.0 <= water_level < 6.5:
        waterphase = 5
    elif 6.5 <= water_level < 7.0:
        waterphase = 6
    else:
        waterphase = 7

    print(f"waterphase: {waterphase}")

    uploaded_file_url = upload_to_drive("snapshot.jpg", "snapshot_uploaded.jpg", PARENT_FOLDER_ID)
    print(f"รูปภาพอัปโหลดสำเร็จ: {uploaded_file_url}")

    return water_level, waterphase, uploaded_file_url

def send_line_notification(water_level, waterphase, image_url):
    now = datetime.now(thai_tz).strftime('%H:%M:%S')
    if waterphase == 1:
        gauging_station = (" สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n "
        " -น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย ")
        danger = ("✅ ปกติ\n"
          "🟢 ยังไม่พบความเสี่ยงจากน้ำท่วมในขณะนี้ "
        )
        shelter_info = " - "
    elif waterphase == 2:
        gauging_station = (" สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n "
        " -น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย ")
        danger = ("⬆️ ระดับน้ำเพิ่มสูงกว่าปกติ\n"
          "🟡โปรดติดตามสถานการณ์อย่างใกล้ชิด "
        )
        shelter_info = " - "
    elif waterphase == 3:
        gauging_station = (" สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n "
        " -น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย ")
        danger = ("⬆️ ระดับน้ำเพิ่มสูงกว่าปกติ\n"
          " 🟡เฝ้าระวังเป็นพิเศษ หากมีฝนตกเพิ่มอาจส่งผลกระทบ "
        )
        shelter_info = " - "
    elif waterphase == 4:
        gauging_station = (" สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n "
        " -น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย ")
        danger =  ("⚠️ เฝ้าระวัง\n"
          " 🔴ประชาชนในพื้นที่เสี่ยงควรเตรียมความพร้อมและติดตามประกาศจากทางการ "
        )
        shelter_info = " - "
    elif waterphase == 5:
        gauging_station = (" สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n "
        " -น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย ")
        danger = "🔴 เตรียมอพยพ"
        shelter_info = " - "
    elif waterphase == 6:
        gauging_station = ( "สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n"
        "- น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย"
    )
        danger = (
        "🔴 อพยพ 🔴\n"
        "🔴 ขอให้ประชาชนในพื้นที่เสี่ยงดำเนินการอพยพทันที"
    )
        shelter_info = (
        "🏠 ศูนย์พักพิง:\n"
        "- โรงเรียนเทศบาล 6 นครเชียงราย โทร. 053-152-153\n"
        "- โรงเรียนเทศบาล 7 ฝั่งหมิ่น โทร. 053-166-956\n"
        "- โรงเรียนอบจ. เชียงราย อาคารอเนกประสงค์ โทร. 053-711-333"
    )

    else:
        gauging_station = (" สถานีวัดระดับน้ำ โรงเรียนเทศบาล 6 นครเชียงราย\n "
        " -น้ำกก ต.ริมกก อ.เมือง จ.เชียงราย ")
        danger = ("🚨 วิกฤต\n"
          " 🔴⚠️ขอให้ประชาชนปฎิบัติตามคำสั่งของเจ้าหน้าที่อย่างเคร่งครัด "
        )
        shelter_info = (
            "🏠 ศูนย์พักพิง:\n"
           "- โรงเรียนเทศบาล 6 นครเชียงราย โทร. 053-152-153\n"
           "- โรงเรียนเทศบาล 7 ฝั่งหมิ่น โทร. 053-166-956\n"
           "- โรงเรียนอบจ. เชียงราย อาคารอเนกประสงค์ โทร. 053-711-333"
        )

    messages = [
        {
            "type": "image",
            "originalContentUrl": image_url,
            "previewImageUrl": image_url,
            "altText": "Water Level Snapshot"
        },
        {
            "type": "text",
            "text": f"🔔 วันที่: {datetime.now(thai_tz).strftime('%d/%m/%Y')}\n🔷 สถานี: {gauging_station}\n🌊 ระดับน้ำ: {water_level} เมตร"
        },
        {
            "type": "text",
            "text": f"📢 สถานะความปลอดภัย: {danger}"
        },
        {
            "type": "text",
            "text": f"📍 ศูนย์บริการ: {shelter_info}"
        },
        {
            "type": "text",
            "text": f"🕒 เวลาที่รายงาน: {now} น."
        }
    ]

    data = {"to": "YOUR-ID", "messages": messages}
    headers = {
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(LINE_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        print(f"[{now}] ส่งการแจ้งเตือนผ่าน LINE สำเร็จ")
    else:
        print(f"[{now}] เกิดข้อผิดพลาด: {response.status_code}, {response.text}")


print("เริ่มการตรวจสอบระดับน้ำ...")

water_level, curphase, image_url = capture_and_process()
prephase = curphase

next_time = datetime.now(thai_tz)

while True:
    water_level, curphase, image_url = capture_and_process()

    if water_level is not None:
        if curphase != prephase:
            send_line_notification(water_level, curphase, image_url)
            prephase = curphase

        next_time = datetime.now(thai_tz) + timedelta(minutes=1)
        print(f"รอจนถึง {next_time.strftime('%H:%M:%S')} เพื่อคำนวณระดับน้ำอีกครั้ง...")

        while datetime.now(thai_tz) < next_time:
            time.sleep(1)
    else:
        print("เกิดข้อผิดพลาดในการดึงภาพ / คำนวณระดับน้ำ ลองใหม่ใน 1 นาที...")
        time.sleep(60)