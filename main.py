from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8s.pt for better accuracy

def preprocess_image(image: Image.Image):
    """Enhance contrast and brightness for low-light images."""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to LAB color space for better contrast adjustment
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    processed_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Apply gamma correction
    gamma = 1.5  # Adjust for brightness
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    processed_img = cv2.LUT(processed_img, look_up_table)
    
    return processed_img

async def detect_people_in_image(image: UploadFile):
    """Detect people in a single image."""
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    processed_image = preprocess_image(pil_image)
    results = model(processed_image)
    
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    num_people = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # Person class in COCO dataset
                num_people += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    _, encoded_img = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(encoded_img).decode('utf-8')
    
    return {"image": img_base64, "count": num_people}

@app.post("/detect/")
async def detect_people(images: list[UploadFile] = File(...)):
    """Detect people in multiple images concurrently."""
    tasks = [detect_people_in_image(image) for image in images]
    results = await asyncio.gather(*tasks)
    
    response_data = []
    for idx, result in enumerate(results):
        response_data.append({
            "filename": images[idx].filename,
            "people_count": result["count"],
            "image": result["image"]
        })
    
    return JSONResponse(content={"detections": response_data})