from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

# ✅ Serve static files (IMPORTANT)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Upload folder
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load model
model = load_model(BASE_DIR / "brain_tumor_model.keras")

classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ---------------- HOME ---------------- #

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="text-align:center; font-family:Arial;">
        <h2>🧠 Brain Tumor Classification</h2>

        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br><br>
            <button type="submit">Upload & Predict</button>
        </form>
    </body>
    </html>
    """

# ---------------- PREDICT ---------------- #

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):

    try:
        filepath = UPLOAD_FOLDER / file.filename

        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Preprocess
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        pred = model.predict(img_array)
        result = classes[np.argmax(pred)]

        # ✅ IMPORTANT: correct static path
        img_url = f"/static/uploads/{file.filename}"

        return f"""
        <html>
        <body style="text-align:center; font-family:Arial;">
            <h2>🧠 Brain Tumor Classification</h2>

            <h3>Prediction: {result}</h3>

            <img src="{img_url}" width="250"><br><br>

            <a href="/">⬅ Go Back</a>
        </body>
        </html>
        """

    except Exception as e:
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>")