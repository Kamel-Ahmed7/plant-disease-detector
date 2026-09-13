from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# السماح للفرونت إند بالاتصال بالـ API بدون مشاكل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل الموديل المجهز من فولدر saved_models
MODEL = tf.keras.models.load_model("../saved_models/potatoes_model.keras")

# أسماء الكلاسات بنفس ترتيب تدريب الموديل
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))  # إرجاع الصورة لأبعاد المدخلات المتوقعة
    image_batch = np.expand_dims(np.array(image), 0)
    return image_batch

@app.get("/ping")
async def ping():
    return "Server is running!"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. قراءة وتحضير الصورة
    image_bytes = await file.read()
    image = read_file_as_image(image_bytes)
    
    # 2. التنبؤ بالموديل
    predictions = MODEL.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    # 3. إرجاع النتيجة كـ JSON
    return {
        'class': predicted_class,
        'confidence': round(confidence * 100, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)