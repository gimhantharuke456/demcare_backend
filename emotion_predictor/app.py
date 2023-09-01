from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

# Load the emotion detection model
model = load_model('emotion_model.h5')
model.make_predict_function()

# Define the emotion labels
emotions = ['Happy', 'Sad', 'Angry', 'Neutral']

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((48, 48))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Route for emotion prediction API
@app.post("/predict")
async def predict_emotion(download_url: str):
    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)[0]
            predicted_emotion = emotions[np.argmax(prediction)]
            return JSONResponse(content={"predicted_emotion": predicted_emotion}, status_code=200)
        else:
            return HTTPException(detail="Failed to download image", status_code=500)
    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
