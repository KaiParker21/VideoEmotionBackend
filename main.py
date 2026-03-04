from fastapi import FastAPI, UploadFile, File
from emotion_model import detect_emotion

app = FastAPI(title="Emotion Detection API")

@app.post("/detect-emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    emotion, confidence = detect_emotion(contents)
    
    return {
        "emotion": emotion,
        "confidence": confidence
    }
